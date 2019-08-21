#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/Detector.h"
#include <TGeoBBox.h>
#include <TGeoBoolNode.h>
#include <vector>

using namespace cms;
using namespace edm;
using namespace std;
using namespace cms::dd;

// These names are defined in the DD4hep source code.
// They are returned by dd4hep::Solid.GetTitle(). If DD4hep changes them,
// the names must be updated here.
static const char* const pseudoTrapName = "pseudotrap";
static const char* const truncTubeName = "trunctube";

DDFilteredView::DDFilteredView(const DDDetector* det, const Volume volume) : registry_(&det->specpars()) {
  it_.emplace_back(Iterator(volume));
}

DDFilteredView::DDFilteredView(const DDCompactView& cpv, const DDFilter& attribute) : registry_(&cpv.specpars()) {
  it_.emplace_back(Iterator(cpv.detector()->worldVolume()));
  DDSpecParRefs refs;
  registry_->filter(refs, attribute);
  mergedSpecifics(refs);
  LogVerbatim("Geometry").log([&refs](auto& log) {
    log << "Filtered DD SpecPar Registry size: " << refs.size() << "\n";
    for (const auto& t : refs) {
      log << "\nRegExps { ";
      for (const auto& ki : t->paths)
        log << ki << " ";
      log << "};\n ";
      for (const auto& kl : t->spars) {
        log << kl.first << " = ";
        for (const auto& kil : kl.second) {
          log << kil << " ";
        }
        log << "\n ";
      }
    }
  });
}

const PlacedVolume DDFilteredView::volume() const { return PlacedVolume(node_); }

const Double_t* DDFilteredView::trans() const { return it_.back().GetCurrentMatrix()->GetTranslation(); }

const Translation DDFilteredView::translation() const {
  const Double_t* translation = it_.back().GetCurrentMatrix()->GetTranslation();
  assert(translation);
  return Translation(translation[0], translation[1], translation[2]);
}

const Double_t* DDFilteredView::rot() const { return it_.back().GetCurrentMatrix()->GetRotationMatrix(); }

const RotationMatrix DDFilteredView::rotation() const {
  const Double_t* rotation = it_.back().GetCurrentMatrix()->GetRotationMatrix();
  if (rotation == nullptr) {
    LogError("DDFilteredView") << "Current node has no valid rotation matrix.";
    return RotationMatrix();
  }

  LogVerbatim("DDFilteredView") << "Rotation matrix components (1st 3) = " << rotation[0] << ", " << rotation[1] << ", "
                                << rotation[2];
  RotationMatrix rotMatrix;
  rotMatrix.SetComponents(rotation[0],
                          rotation[1],
                          rotation[2],
                          rotation[3],
                          rotation[4],
                          rotation[5],
                          rotation[6],
                          rotation[7],
                          rotation[8]);
  return rotMatrix;
}

void DDFilteredView::rot(dd4hep::Rotation3D& matrixOut) const {
  const Double_t* rotation = it_.back().GetCurrentMatrix()->GetRotationMatrix();
  if (rotation == nullptr) {
    LogError("DDFilteredView") << "Current node has no valid rotation matrix.";
    return;
  }
  LogVerbatim("DDFilteredView") << "Rotation matrix components (1st 3) = " << rotation[0] << ", " << rotation[1] << ", "
                                << rotation[2];
  matrixOut.SetComponents(rotation[0],
                          rotation[1],
                          rotation[2],
                          rotation[3],
                          rotation[4],
                          rotation[5],
                          rotation[6],
                          rotation[7],
                          rotation[8]);
}

void DDFilteredView::mergedSpecifics(DDSpecParRefs const& specs) {
  for (const auto& i : specs) {
    for (const auto& j : i->paths) {
      vector<string_view> toks = split(j, "/");
      auto const& filter = find_if(begin(filters_), end(filters_), [&](auto const& f) {
        auto const& k = find(begin(f->keys), end(f->keys), toks.front());
        if (k != end(f->keys)) {
          currentFilter_ = f.get();
          return true;
        }
        return false;
      });
      if (filter == end(filters_)) {
        filters_.emplace_back(unique_ptr<Filter>(new Filter{{toks.front()}, nullptr, nullptr}));
        currentFilter_ = filters_.back().get();
      }
      // all next levels
      for (size_t pos = 1; pos < toks.size(); ++pos) {
        if (currentFilter_->next != nullptr) {
          currentFilter_ = currentFilter_->next.get();
          auto const& l = find(begin(currentFilter_->keys), end(currentFilter_->keys), toks[pos]);
          if (l == end(currentFilter_->keys)) {
            currentFilter_->keys.emplace_back(toks[pos]);
          }
        } else {
          currentFilter_->next.reset(new Filter{{toks[pos]}, nullptr, currentFilter_});
        }
      }
    }
  }
}

bool DDFilteredView::firstChild() {
  if (it_.empty()) {
    LogVerbatim("DDFilteredView") << "Iterator vector has zero size.";
    return false;
  }
  it_.back().SetType(0);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (accept(node->GetVolume()->GetName())) {
      addPath(node);
      return true;
    }
  }
  LogVerbatim("DDFilteredView") << "Search for first child failed.";
  return false;
}

bool DDFilteredView::firstSibling() {
  if (it_.empty() || currentFilter_ == nullptr)
    return false;
  if (next(0) == false)
    return false;
  it_.emplace_back(Iterator(it_.back()));
  it_.back().SetType(1);
  if (currentFilter_ != nullptr && currentFilter_->next != nullptr)
    currentFilter_ = currentFilter_->next.get();
  else
    return false;
  do {
    if (accepted(currentFilter_->keys, node_->GetVolume()->GetName())) {
      addNode(node_);
      return true;
    }
  } while ((node_ = it_.back().Next()));

  return false;
}

bool DDFilteredView::nextSibling() {
  if (it_.empty() || currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  unCheckNode();
  do {
    if (accepted(currentFilter_->keys, node_->GetVolume()->GetName())) {
      addNode(node_);
      return true;
    }
  } while ((node_ = it_.back().Next()));

  return false;
}

bool DDFilteredView::sibling() {
  if (it_.empty() || currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (accepted(currentFilter_->keys, node->GetVolume()->GetName())) {
      addNode(node);
      return true;
    }
  }
  return false;
}

bool DDFilteredView::siblingNoCheck() {
  if (it_.empty() || currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (accepted(currentFilter_->keys, node->GetVolume()->GetName())) {
      node_ = node;
      return true;
    }
  }
  return false;
}

bool DDFilteredView::checkChild() {
  if (it_.empty() || currentFilter_ == nullptr)
    return false;
  it_.back().SetType(1);
  Node* node = nullptr;
  while ((node = it_.back().Next())) {
    if (accepted(currentFilter_->keys, node->GetVolume()->GetName())) {
      return true;
    }
  }
  return false;
}

bool DDFilteredView::parent() {
  if (it_.empty() || currentFilter_ == nullptr)
    return false;
  up();
  it_.back().SetType(0);
  it_.back().Skip();

  return true;
}

bool DDFilteredView::next(int type) {
  if (it_.empty())
    return false;
  it_.back().SetType(type);
  Node* node = nullptr;
  if ((node = it_.back().Next())) {
    node_ = node;
    return true;
  } else
    return false;
}

void DDFilteredView::down() {
  if (it_.empty() || currentFilter_ == nullptr)
    return;
  it_.emplace_back(Iterator(it_.back()));
  next(0);
  if (currentFilter_->next)
    currentFilter_ = currentFilter_->next.get();
}

void DDFilteredView::up() {
  if (it_.empty() || currentFilter_ == nullptr)
    return;
  it_.pop_back();
  if (currentFilter_->up)
    currentFilter_ = currentFilter_->up;
}

bool DDFilteredView::accept(std::string_view name) {
  bool result = false;
  for (const auto& it : filters_) {
    currentFilter_ = it.get();
    result = accepted(currentFilter_->keys, name);
    if (result)
      return result;
  }
  return result;
}

// FIXME: obsolete
vector<double> DDFilteredView::extractParameters() const {
  Volume currVol = node_->GetVolume();
  if (currVol->GetShape()->IsA() == TGeoBBox::Class()) {
    const TGeoBBox* box = static_cast<const TGeoBBox*>(currVol->GetShape());
    return {box->GetDX(), box->GetDY(), box->GetDZ()};
  } else if (currVol->GetShape()->IsA() == TGeoCompositeShape::Class()) {
    const TGeoCompositeShape* shape = static_cast<const TGeoCompositeShape*>(currVol->GetShape());
    const TGeoBoolNode* boolean = shape->GetBoolNode();
    while (boolean->GetLeftShape()->IsA() != TGeoBBox::Class()) {
      boolean = static_cast<const TGeoCompositeShape*>(boolean->GetLeftShape())->GetBoolNode();
    }
    const TGeoBBox* box = static_cast<const TGeoBBox*>(boolean->GetLeftShape());
    return {box->GetDX(), box->GetDY(), box->GetDZ()};
  } else
    return {1, 1, 1};
}

const std::vector<double> DDFilteredView::parameters() const {
  Volume currVol = node_->GetVolume();
  return currVol.solid().dimensions();
}

const DDSolidShape DDFilteredView::shape() const {
  //FIXME
  return DDSolidShape::dd_not_init;
}

double DDFilteredView::getDouble(std::string_view key) const {
  //FIXME
  return 0;
}

std::string DDFilteredView::getString(std::string_view key) const {
  //FIXME
  return std::string("none");
}

bool DDFilteredView::addPath(Node* const node) {
  assert(registry_);
  node_ = node;
  nodes_.tags.clear();
  nodes_.offsets.clear();
  nodes_.copyNos.clear();
  bool result(false);

  int level = it_.back().GetLevel();
  for (int nit = level; nit > 0; --nit) {
    for_each(begin(registry_->specpars), end(registry_->specpars), [&](auto const& i) {
      auto k = find_if(begin(i.second.paths), end(i.second.paths), [&](auto const& j) {
        return (compareEqual(it_.back().GetNode(nit)->GetVolume()->GetName(), *begin(split(realTopName(j), "/"))) &&
                (i.second.hasValue("CopyNoTag") || i.second.hasValue("CopyNoOffset")));
      });
      if (k != end(i.second.paths)) {
        nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
        nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
        nodes_.copyNos.emplace_back(it_.back().GetNode(nit)->GetNumber());
        result = true;
      }
    });
  }
  return result;
}

bool DDFilteredView::addNode(Node* const node) {
  assert(registry_);
  node_ = node;
  bool result(false);
  for_each(begin(registry_->specpars), end(registry_->specpars), [&](auto const& i) {
    auto k = find_if(begin(i.second.paths), end(i.second.paths), [&](auto const& j) {
      return (compareEqual(node_->GetVolume()->GetName(), *begin(split(realTopName(j), "/"))) &&
              (i.second.hasValue("CopyNoTag") || i.second.hasValue("CopyNoOffset")));
    });
    if (k != end(i.second.paths)) {
      nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
      nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
      nodes_.copyNos.emplace_back(node_->GetNumber());
      result = true;
    }
  });
  return result;
}

void DDFilteredView::unCheckNode() {
  nodes_.tags.pop_back();
  nodes_.offsets.pop_back();
  nodes_.copyNos.pop_back();
}

const TClass* DDFilteredView::getShape() const {
  Volume currVol = node_->GetVolume();
  return (currVol->GetShape()->IsA());
}

bool DDFilteredView::isABox() const { return (getShape() == TGeoBBox::Class()); }

bool DDFilteredView::isAConeSeg() const { return (getShape() == TGeoConeSeg::Class()); }

bool DDFilteredView::isAPseudoTrap() const {
  LogVerbatim("DDFilteredView") << "Shape is a " << solid()->GetTitle() << ".";
  return (strcmp(solid()->GetTitle(), pseudoTrapName) == 0);
}

bool DDFilteredView::isATrapezoid() const { return (getShape() == TGeoTrap::Class()); }

bool DDFilteredView::isATruncTube() const {
  LogVerbatim("DDFilteredView") << "Shape is a " << solid()->GetTitle() << ".";
  return (strcmp(solid()->GetTitle(), truncTubeName) == 0);
}

bool DDFilteredView::isATubeSeg() const { return (getShape() == TGeoTubeSeg::Class()); }

std::string_view DDFilteredView::name() const { return (volume().volume().name()); }

dd4hep::Solid DDFilteredView::solid() const { return (volume().volume().solid()); }

unsigned short DDFilteredView::copyNum() const { return (volume().copyNumber()); }

std::string_view DDFilteredView::materialName() const { return (volume().material().name()); }
