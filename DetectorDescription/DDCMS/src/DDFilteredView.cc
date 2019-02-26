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

DDFilteredView::DDFilteredView(const DDDetector* det)
  : registry_(&det->specpars()) {
  dd4hep::DetElement world = det->description()->world();
  parents_.emplace_back(DDExpandedNode(world.volume(), DDTranslation(), DDRotationMatrix(), 1, 0));
  topVolume_ = world.volume();
  TGeoIterator next(topVolume_);
  next.SetType(0); // 0: DFS; 1: BFS
}

const DDVolume&
DDFilteredView::volume() const {
  return parents_.back().volume;
}

const DDTranslation&
DDFilteredView::translation() const {
  return parents_.back().trans;
}

const DDRotationMatrix&
DDFilteredView::rotation() const {
  return parents_.back().rot;
}

void
DDFilteredView::mergedSpecifics(DDSpecParRefs const& refs) {
  for(const auto& i : refs) {
    auto tops = vPathsTo(*i, 1);
    //    auto tops = vPathsTo(*i.second, 1);
    topNodes_.insert(end(topNodes_), begin(tops), end(tops));
  }
}

bool
DDFilteredView::firstChild() {
  return false;
}

bool
DDFilteredView::nextSibling() {
  return false;
}

bool
DDFilteredView::next() {
  return false;
}

const DDGeoHistory&
DDFilteredView::geoHistory() const {
  return parents_;
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
  for(auto const& i : names) {
    if(accepted(i, node)) {
      return true;
    }
  }
  return false;
}

bool
DDFilteredView::acceptedM(vector<string_view>& names, string_view node) const {
  auto itr = find_if(names.begin(), names.end(), [ & ]( const auto& i){ return accepted(i, node); });
  if(itr != names.end() && (!isRegex(*itr))) {
    names.erase(itr);
    return true;
  }
  return false;
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
DDFilteredView::checkPath(string_view path, TGeoNode *node) {
  assert(registry_);
  node_ = node;
  nodes.tags.clear();
  nodes.offsets.clear();
  nodes.copyNos.clear();
  
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
  	  nodes.tags.emplace_back(i.second.dblValue("CopyNoTag"));
  	  nodes.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
  	  nodes.copyNos.emplace_back(copyNo(rv));
  	  result = true;
  	}
      });
  }
  return result;
}

bool
DDFilteredView::checkNode(TGeoNode *node) {
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
	nodes.tags.emplace_back(i.second.dblValue("CopyNoTag"));
	nodes.offsets.emplace_back(i.second.dblValue("CopyNoOffset"));
	nodes.copyNos.emplace_back(copyNo(node_->GetName()));
	result = true;
      }
    });
  return result;
}

void
DDFilteredView::unCheckNode() {
   nodes.tags.pop_back();
   nodes.offsets.pop_back();
   nodes.copyNos.pop_back();
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

vector<string_view>
DDFilteredView::vPathsTo(const DDSpecPar& specpar, unsigned int level) const {
  vector<string_view> result;
  for(auto const& i : specpar.paths) {
    vector<string_view> toks = split(i, "/");
    if(level == toks.size())
      result.emplace_back(realTopName(i));
  }
  return result;
}

vector<string_view>
DDFilteredView::tails(const vector<string_view>& fullPath) const {
  vector<string_view> result;
  for(auto const& v : fullPath) {
    auto found = v.find_last_of("/");
    if(found != v.npos) {
      result.emplace_back(v.substr(found + 1));
    }
  }
  return result;
}
