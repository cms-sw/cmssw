#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/TrackerShapeToBounds.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include <DD4hep/DD4hepUnits.h>

#include <cfloat>
#include <vector>
#include <string>

namespace {

  const std::string strue("true");

  template <typename DDView>
  double getDouble(const char* s, DDView const& ev) {
    DDValue val(s);
    std::vector<const DDsvalues_type*> result;
    ev.specificsV(result);
    std::vector<const DDsvalues_type*>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it) {
      foundIt = DDfetch(*it, val);
      if (foundIt)
        break;
    }
    if (foundIt) {
      const std::vector<std::string>& temp = val.strings();
      if (temp.size() != 1) {
        throw cms::Exception("Configuration") << "I need 1 " << s << " tags";
      }
      return double(::atof(temp[0].c_str()));
    }
    return 0;
  }

  template <typename DDView>
  std::string getString(const char* s, DDView const& ev) {
    DDValue val(s);
    std::vector<const DDsvalues_type*> result;
    ev.specificsV(result);
    std::vector<const DDsvalues_type*>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it) {
      foundIt = DDfetch(*it, val);
      if (foundIt)
        break;
    }
    if (foundIt) {
      const std::vector<std::string>& temp = val.strings();
      if (temp.size() != 1) {
        throw cms::Exception("Configuration") << "I need 1 " << s << " tags";
      }
      return temp[0];
    }
    return "NotFound";
  }
}  // namespace

/**
 * What to do in the destructor?
 * destroy all the daughters!
 */
GeometricTimingDet::~GeometricTimingDet() { deleteComponents(); }

GeometricTimingDet::GeometricTimingDet(DDFilteredView* fv, GeometricTimingEnumType type)
    :  //
      // Set by hand the ddd_
      //
      trans_(fv->translation()),
      phi_(trans_.Phi()),
      rho_(trans_.Rho()),
      rot_(fv->rotation()),
      shape_(cms::dd::name_from_value(cms::LegacySolidShapeMap, fv->shape())),
      ddname_(fv->name()),
      type_(type),
      params_(fv->parameters()),
      radLength_(getDouble("TrackerRadLength", *fv)),
      xi_(getDouble("TrackerXi", *fv)),
      pixROCRows_(getDouble("PixelROCRows", *fv)),
      pixROCCols_(getDouble("PixelROCCols", *fv)),
      pixROCx_(getDouble("PixelROC_X", *fv)),
      pixROCy_(getDouble("PixelROC_Y", *fv)),
      stereo_(getString("TrackerStereoDetectors", *fv) == strue),
      siliconAPVNum_(getDouble("SiliconAPVNumber", *fv)) {
  const DDFilteredView::nav_type& nt = fv->navPos();
  ddd_ = nav_type(nt.begin(), nt.end());
}

GeometricTimingDet::GeometricTimingDet(cms::DDFilteredView* fv, GeometricTimingEnumType type)
    : trans_(fv->translation() / dd4hep::mm),
      rot_(fv->rotation()),
      shape_(fv->shape()),
      ddname_(fv->name()),
      type_(type),
      params_(fv->parameters()),
      radLength_(fv->get<double>("TrackerRadLength")),
      xi_(fv->get<double>("TrackerXi")),
      pixROCRows_(fv->get<double>("PixelROCRows")),
      pixROCCols_(fv->get<double>("PixelROCCols")),
      pixROCx_(fv->get<double>("PixelROC_X")),
      pixROCy_(fv->get<double>("PixelROC_Y")),
      stereo_(fv->get<std::string_view>("TrackerStereoDetectors") == strue),
      siliconAPVNum_(fv->get<double>("SiliconAPVNumber")) {
  phi_ = trans_.Phi();
  rho_ = trans_.Rho();
  for (size_t pit = 0; pit < params_.size(); pit++) {
    params_[pit] = params_[pit] / dd4hep::mm;
  }
  //
  // Not navPos(), as not properly working for DD4hep and not used
  //
  ddd_ = nav_type(fv->copyNos().size(), 0);
}

// PGeometricTimingDet is persistent version... make it... then come back here and make the
// constructor.
GeometricTimingDet::GeometricTimingDet(const PGeometricTimingDet::Item& onePGD, GeometricTimingEnumType type)
    : trans_(onePGD.x_, onePGD.y_, onePGD.z_),
      phi_(onePGD.phi_),  //_trans.Phi()),
      rho_(onePGD.rho_),  //_trans.Rho()),
      rot_(onePGD.a11_,
           onePGD.a12_,
           onePGD.a13_,
           onePGD.a21_,
           onePGD.a22_,
           onePGD.a23_,
           onePGD.a31_,
           onePGD.a32_,
           onePGD.a33_),
      shape_(cms::dd::name_from_value(cms::LegacySolidShapeMap, static_cast<LegacySolidShape>(onePGD.shape_))),
      ddd_(),
      ddname_(onePGD.name_),  //, "fromdb");
      type_(type),
      params_(),
      geographicalID_(onePGD.geographicalID_),
      radLength_(onePGD.radLength_),
      xi_(onePGD.xi_),
      pixROCRows_(onePGD.pixROCRows_),
      pixROCCols_(onePGD.pixROCCols_),
      pixROCx_(onePGD.pixROCx_),
      pixROCy_(onePGD.pixROCy_),
      stereo_(onePGD.stereo_),
      siliconAPVNum_(onePGD.siliconAPVNum_) {
  if (onePGD.shape_ == 1 || onePGD.shape_ == 3) {  //The parms vector is neede only in the case of box or trap shape
    params_.reserve(11);
    params_.emplace_back(onePGD.params_0);
    params_.emplace_back(onePGD.params_1);
    params_.emplace_back(onePGD.params_2);
    params_.emplace_back(onePGD.params_3);
    params_.emplace_back(onePGD.params_4);
    params_.emplace_back(onePGD.params_5);
    params_.emplace_back(onePGD.params_6);
    params_.emplace_back(onePGD.params_7);
    params_.emplace_back(onePGD.params_8);
    params_.emplace_back(onePGD.params_9);
    params_.emplace_back(onePGD.params_10);
  }

  ddd_.reserve(onePGD.numnt_);
  ddd_.emplace_back(onePGD.nt0_);
  ddd_.emplace_back(onePGD.nt1_);
  ddd_.emplace_back(onePGD.nt2_);
  ddd_.emplace_back(onePGD.nt3_);
  if (onePGD.numnt_ > 4) {
    ddd_.emplace_back(onePGD.nt4_);
    if (onePGD.numnt_ > 5) {
      ddd_.emplace_back(onePGD.nt5_);
      if (onePGD.numnt_ > 6) {
        ddd_.emplace_back(onePGD.nt6_);
        if (onePGD.numnt_ > 7) {
          ddd_.emplace_back(onePGD.nt7_);
          if (onePGD.numnt_ > 8) {
            ddd_.emplace_back(onePGD.nt8_);
            if (onePGD.numnt_ > 9) {
              ddd_.emplace_back(onePGD.nt9_);
              if (onePGD.numnt_ > 10) {
                ddd_.emplace_back(onePGD.nt10_);
              }
            }
          }
        }
      }
    }
  }
}

GeometricTimingDet::ConstGeometricTimingDetContainer GeometricTimingDet::deepComponents() const {
  //
  // iterate on all the components ;)
  //
  ConstGeometricTimingDetContainer temp;
  deepComponents(temp);
  return temp;
}

void GeometricTimingDet::deepComponents(ConstGeometricTimingDetContainer& cont) const {
  if (isLeaf()) {
    cont.emplace_back(this);
  } else
    std::for_each(
        container_.begin(), container_.end(), [&](const GeometricTimingDet* iDet) { iDet->deepComponents(cont); });
}

void GeometricTimingDet::addComponents(GeometricTimingDetContainer const& cont) {
  container_.reserve(container_.size() + cont.size());
  std::copy(cont.begin(), cont.end(), back_inserter(container_));
}

void GeometricTimingDet::addComponents(ConstGeometricTimingDetContainer const& cont) {
  container_.reserve(container_.size() + cont.size());
  std::copy(cont.begin(), cont.end(), back_inserter(container_));
}

void GeometricTimingDet::addComponent(GeometricTimingDet* det) { container_.emplace_back(det); }

namespace {
  struct Deleter {
    void operator()(GeometricTimingDet const* det) const { delete const_cast<GeometricTimingDet*>(det); }
  };
}  // namespace

void GeometricTimingDet::deleteComponents() {
  std::for_each(container_.begin(), container_.end(), Deleter());
  container_.clear();
}

GeometricTimingDet::Position GeometricTimingDet::positionBounds() const {
  Position pos(geant_units::operators::convertMmToCm(trans_.x()),
               geant_units::operators::convertMmToCm(trans_.y()),
               geant_units::operators::convertMmToCm(trans_.z()));
  return pos;
}

GeometricTimingDet::Rotation GeometricTimingDet::rotationBounds() const {
  Translation x, y, z;
  rot_.GetComponents(x, y, z);
  Rotation rotation(float(x.X()),
                    float(x.Y()),
                    float(x.Z()),
                    float(y.X()),
                    float(y.Y()),
                    float(y.Z()),
                    float(z.X()),
                    float(z.Y()),
                    float(z.Z()));
  return rotation;
}

std::unique_ptr<Bounds> GeometricTimingDet::bounds() const {
  const std::vector<double>& par = params_;
  TrackerShapeToBounds shapeToBounds;
  return std::unique_ptr<Bounds>(shapeToBounds.buildBounds(shape_, par));
}
