#ifndef DETECTOR_DESCRIPTION_REGRESSION_TEST_TINY_DOM2_H
#define DETECTOR_DESCRIPTION_REGRESSION_TEST_TINY_DOM2_H

#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include "DetectorDescription/RegressionTest/src/TagName.h"

class AnotherDummy2 {};

using AttList2 = std::map<TagName, TagName>;
using Node2 = std::pair<TagName, AttList2>;
using TinyDom2 = math::Graph<Node2, AnotherDummy2>;
using TinyDom2Walker = math::GraphWalker<Node2, AnotherDummy2>;
  
void TinyDom2PrettyPrint(std::ostream &, const TinyDom2 &);

#endif
