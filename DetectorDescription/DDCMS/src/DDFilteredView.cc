#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"
#include <TGeoBBox.h>
#include <TGeoBoolNode.h>
#include <experimental/iterator>
#include <functional>
#include <regex>
#include <vector>

using namespace cms;
using namespace std;

DDFilteredView::DDFilteredView(const DDDetector* det, const Volume volume)
  : registry_(&det->specpars()) {
  it_.emplace_back(TGeoIterator(volume));
}

const PlacedVolume
DDFilteredView::volume() const {
  return PlacedVolume(node_);
}

const Double_t*
DDFilteredView::trans() const {
  return it_.back().GetCurrentMatrix()->GetTranslation();
}

const Double_t*
DDFilteredView::rot() const {
  return it_.back().GetCurrentMatrix()->GetRotationMatrix();
}

void
DDFilteredView::mergedSpecifics(DDSpecParRefs const& specs) {
  for(const auto& i : specs) {
    for(const auto& j : i->paths) {
      vector<string_view> toks = split(j, "/");
      auto const& filter = find_if(begin(filters_), end(filters_), [&](auto const& f) {
	  auto const& k = find(begin(f->keys), end(f->keys), toks.front());
	  if(k != end(f->keys)) {
	    currentFilter_ = f.get();
	    return true;
	  }
	  return false;
	});
      if(filter == end(filters_)) {
	filters_.emplace_back(unique_ptr<Filter>(new Filter{{toks.front()}, nullptr, nullptr}));
	currentFilter_ = filters_.back().get();
      }
      // all next levels
      for(size_t pos = 1; pos < toks.size(); ++pos) {
	if(currentFilter_->next != nullptr) {
	  currentFilter_ = currentFilter_->next.get();
	  auto const& l = find(begin(currentFilter_->keys), end(currentFilter_->keys), toks[pos]);
	  if(l == end(currentFilter_->keys)) {
	    currentFilter_->keys.emplace_back(toks[pos]);
	  }
	} else {
	  currentFilter_->next.reset(new Filter{{toks[pos]}, nullptr, currentFilter_});
	}
      }
    }
  }
}

bool
DDFilteredView::firstChild() {
  it_.back().SetType(0);
  TGeoNode *node = nullptr;
  while((node = it_.back().Next())) {
    if(accepted(noNamespace(node->GetVolume()->GetName()))) {
      TString path;
      it_.back().GetPath(path);
      addPath(path, node);

      return true;
    }
  }
  return false;
}

bool
DDFilteredView::firstSibling() {
  next(0);
  it_.emplace_back(TGeoIterator(it_.back()));
  it_.back().SetType(1);
  if(currentFilter_->next)
    currentFilter_ = currentFilter_->next.get();
  else
    return false;
  do {
    if(accepted(currentFilter_->keys, noNamespace(node_->GetVolume()->GetName()))) {
      addNode(node_);
      return true;
    }
  } while((node_ = it_.back().Next()));

  return false;
}

bool
DDFilteredView::nextSibling() {
  it_.back().SetType(1);
  unCheckNode();
  do {
    if(accepted(currentFilter_->keys, noNamespace(node_->GetVolume()->GetName()))) {
      addNode(node_);
      return true;
    }
  } while((node_ = it_.back().Next()));

  return false;
}

bool
DDFilteredView::sibling() {
  it_.back().SetType(1);
  TGeoNode *node = nullptr;
  while((node = it_.back().Next())) {
    if(accepted(currentFilter_->keys, noNamespace(node->GetVolume()->GetName()))) {
      addNode(node);
      return true;
    }
  }
  return false;
}

bool
DDFilteredView::siblingNoCheck() {
  it_.back().SetType(1);
  TGeoNode *node = nullptr;
  while((node = it_.back().Next())) {
    if(accepted(currentFilter_->keys, noNamespace(node->GetVolume()->GetName()))) {
      //addNode(node);
      node_ = node;
      return true;
    }
  }
  return false;
}

bool
DDFilteredView::checkChild() {
  it_.back().SetType(1);
  TGeoNode *node = nullptr;
  while((node = it_.back().Next())) {
    if(accepted(currentFilter_->keys, noNamespace(node->GetVolume()->GetName()))) {
      return true;
    }
  }
  return false;
}

bool
DDFilteredView::parent() {
  up();
  it_.back().SetType(0);
  it_.back().Skip();
  
  return true;
}

bool
DDFilteredView::next(int type) {
  it_.back().SetType(type);
  TGeoNode *node = nullptr;
  if((node = it_.back().Next())) {
    node_ = node;
    return true;
  }
  else
    return false;
}

void
DDFilteredView::down() {
  it_.emplace_back(TGeoIterator(it_.back()));
  next(0);
  if(currentFilter_->next)
    currentFilter_ = currentFilter_->next.get();
}

void
DDFilteredView::up() {
  it_.pop_back();
  if(currentFilter_->up)
    currentFilter_ = currentFilter_->up;
}

bool
DDFilteredView::acceptRegex(string_view name, string_view node) const {
  if(!isRegex(name) && !isRegex(node)) {
    return (name == node);
  } else {
    if(isRegex(name)) {
      regex pattern({name.data(), name.size()});
      return regex_search(begin(node), end(node), pattern);
    } else if(isRegex(node)) {
      regex pattern({node.data(), node.size()});
      return regex_search(begin(name), end(name), pattern);
    }
  }
  return false;
}

bool
DDFilteredView::accepted(std::string_view name) {
  bool result = false;
  for(const auto& it : filters_) {
    currentFilter_ = it.get();
    result = accepted(currentFilter_->keys, name);
    if(result)
      return result;
  }
  return result; 
}

bool
DDFilteredView::accepted(string_view name, string_view node) const {
  if(!isRegex(name)) {
    return (name == node);
  } else {
    regex pattern({name.data(), name.size()});
    return regex_search(begin(node), end(node), pattern);
  }
}

bool
DDFilteredView::accepted(vector<string_view> const& names, string_view node) const {
  return (find_if(begin(names), end(names),
		  [&](const auto& n) -> bool { return accepted(n, node); })
	  != end(names));
}

string_view
DDFilteredView::realTopName(string_view input) const {
  string_view v = input;
  auto first = v.find_first_of("//");
  v.remove_prefix(min(first+2, v.size()));
  return v;
}

string_view
DDFilteredView::noNamespace(string_view input) const {
  string_view v = input;
  auto first = v.find_first_of(":");
  v.remove_prefix(min(first+1, v.size()));
  return v;
}

string_view
DDFilteredView::noCopyNo(string_view input) const {
  string_view v = input;
  auto last = v.find_last_of("_");
  v.remove_suffix(v.size() - min(last, v.size()));
  return v;
}

int
DDFilteredView::copyNo(string_view input) const {
  string_view v = input;
  auto last = v.find_last_of("_");
  v.remove_prefix(min(last+1, v.size()));
  
  return stoi({v.data(), v.size()});
}

vector<double>
DDFilteredView::extractParameters() const {
  Volume volume = node_->GetVolume();
  if(volume->GetShape()->IsA() == TGeoBBox::Class()) {
    const TGeoBBox* box = static_cast<const TGeoBBox*>(volume->GetShape());
    return {box->GetDX(), box->GetDY(), box->GetDZ()}; 
  }
  else if(volume->GetShape()->IsA() == TGeoCompositeShape::Class()) {
    const TGeoCompositeShape* shape = static_cast<const TGeoCompositeShape*>(volume->GetShape());
    const TGeoBoolNode* boolean = shape->GetBoolNode();
    while(boolean->GetLeftShape()->IsA() != TGeoBBox::Class()) {
      boolean = static_cast<const TGeoCompositeShape*>(boolean->GetLeftShape())->GetBoolNode();
    }
    const TGeoBBox* box = static_cast<const TGeoBBox*>(boolean->GetLeftShape());
    return {box->GetDX(), box->GetDY(), box->GetDZ()}; 
  } else
    return {1, 1, 1};
}

bool
DDFilteredView::isRegex(string_view input) const {
  return ((contains(input, "*") != -1) ||
	  (contains(input, ".") != -1));
}

int
DDFilteredView::contains(string_view input, string_view needle) const {
  auto const& it = search(begin(input), end(input),
			  boyer_moore_searcher(begin(needle), end(needle)));
  if(it != end(input)) {
    return (it - begin(input));
  }
  return -1;
}

bool
DDFilteredView::addPath(string_view path, TGeoNode* const node) {
  assert(registry_);
  node_ = node;
  nodes_.tags.clear();
  nodes_.offsets.clear();
  nodes_.copyNos.clear();
  
  bool result(false);
  auto v = split(path, "/");
  transform(v.begin(), v.end(), v.begin(),
	    [&](string_view s) -> string_view { return noNamespace(s); });
  
  for(auto const& rv : v) {
    for_each(begin(registry_->specpars), end(registry_->specpars), [&](auto const& i) {
  	auto k = find_if(begin(i.second.paths), end(i.second.paths),[&](auto const& j) {
  	    return (acceptRegex(noCopyNo(rv), *begin(split(realTopName(j), "/"))) &&
  		    (i.second.hasValue("CopyNoTag") ||
  		     i.second.hasValue("CopyNoOffset")));
  	  });
  	if(k != end(i.second.paths)) {
  	  nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
  	  nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
  	  nodes_.copyNos.emplace_back(copyNo(rv));
  	  result = true;
  	}
      });
  }
  return result;
}

bool
DDFilteredView::addNode(TGeoNode* const node) {
  assert(registry_);
  node_ = node;
  bool result(false);
  for_each(begin(registry_->specpars), end(registry_->specpars), [&](auto const& i) {
      auto k = find_if(begin(i.second.paths), end(i.second.paths),[&](auto const& j) {
	  return (acceptRegex(noCopyNo(noNamespace(node_->GetName())), *begin(split(realTopName(j), "/"))) &&
		  (i.second.hasValue("CopyNoTag") ||
		   i.second.hasValue("CopyNoOffset")));
	});
      if(k != end(i.second.paths)) {
	nodes_.tags.emplace_back(i.second.dblValue("CopyNoTag"));
	nodes_.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
	nodes_.copyNos.emplace_back(copyNo(node_->GetName()));
	result = true;
      }
    });
  return result;
}

void
DDFilteredView::unCheckNode() {
   nodes_.tags.pop_back();
   nodes_.offsets.pop_back();
   nodes_.copyNos.pop_back();
}

vector<string_view>
DDFilteredView::split(string_view str, const char* delims) const {
  vector<string_view> ret;
  
  string_view::size_type start = 0;
  auto pos = str.find_first_of(delims, start);
  while(pos != string_view::npos) {
    if(pos != start) {
      ret.emplace_back(str.substr(start, pos - start));
    }
    start = pos + 1;
    pos = str.find_first_of(delims, start);
  }
  if(start < str.length())
    ret.emplace_back(str.substr(start, str.length() - start));
  return ret;
}
