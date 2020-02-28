#include "DetectorDescription/RegressionTest/src/TinyDomTest2.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DetectorDescription/RegressionTest/src/TagName.h"

#include <iostream>
#include <utility>

using std::cout;
using std::endl;
using std::vector;

TinyDomTest2::TinyDomTest2(const TinyDom2& td2) : dom_(td2) {}

unsigned int TinyDomTest2::allNodes(const Node2& n2, vector<const AttList2*>& at2) {
  TinyDom2::const_adj_iterator it = dom_.begin();
  cout << "Size of graph: " << TinyDomTest2::dom_.size() << endl;
  while (it++ != dom_.end()) {
    if (n2.first.sameName(dom_.nodeData(it).first)) {
      at2.emplace_back(&(dom_.nodeData(it).second));
    }
  }
  return at2.size();
}
