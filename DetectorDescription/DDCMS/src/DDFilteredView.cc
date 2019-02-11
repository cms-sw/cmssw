#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include <TGeoBBox.h>
#include <TGeoBoolNode.h>
#include <experimental/iterator>
#include <regex>
#include <vector>

using namespace cms;
using namespace std;

DDFilteredView::DDFilteredView(DDVolume const& volume,
			       DDTranslation const& trans,
			       DDRotationMatrix const& rot) {
  parents_.emplace_back(DDExpandedNode(volume, trans, rot, 1, 0));
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
DDFilteredView::mergedSpecifics(DDSpecParRefMap const& refs) {
  for(const auto& i : refs) {
    auto tops = i.second->vPathsTo(1);
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
DDFilteredView::accepted(string_view name, string_view node) const {
  if(!isRegex(name)) {
    return (name == node);
  } else {
    regex pattern({name.data(), name.size()});
    return regex_search(begin(node), end(node), pattern);
  }
}

bool
DDFilteredView::accepted(vector<string_view> names, string_view node) const {
  for(auto const i : names) {
    if(accepted(i, node))
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

vector<double>
DDFilteredView::extractParameters(Volume volume) const {
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
	  (contains(input, ".") != -1) ||
	  (contains(input, "_") != -1) ||
	  (contains(input, "|") != -1));
}

int
DDFilteredView::contains(string_view in, string_view needle) const {
  auto it = search(begin(in), end(in),
		   boyer_moore_searcher(begin(needle), end(needle)));
  if(it != end(in)) {
    return (it - begin(in));
  }
  return -1;
}
