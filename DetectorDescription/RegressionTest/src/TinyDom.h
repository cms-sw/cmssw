#ifndef DETECTOR_DESCRIPTION_REGRESSION_TEST_TINY_DOM_H
#define DETECTOR_DESCRIPTION_REGRESSION_TEST_TINY_DOM_H

#include <iostream>
#include <map>
#include <string>

#include "DetectorDescription/RegressionTest/src/TagName.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"

class AnotherDummy {};

using NodeName = TagName;
using AttName = TagName;
using AttValue = TagName;
using AttList = std::map<AttName,AttValue>;
using TinyDom = math::Graph<NodeName, AttList>;
using TinyDomWalker = math::GraphWalker<NodeName, AttList>;

void TinyDomPrettyPrint(std::ostream &, const TinyDom &);

#endif
