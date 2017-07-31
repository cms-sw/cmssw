#ifndef DETECTOR_DESCRIPTION_REGRESSION_TEST_TINY_DOM_H
#define DETECTOR_DESCRIPTION_REGRESSION_TEST_TINY_DOM_H

#include <iostream>
#include <map>
#include <string>

#include "DetectorDescription/RegressionTest/src/TagName.h"
#include "Utilities/General/interface/Graph.h"
#include "Utilities/General/interface/GraphWalker.h"

class AnotherDummy {};

using NodeName = TagName;
using AttName = TagName;
using AttValue = TagName;
using AttList = std::map<AttName,AttValue>;
using TinyDom = cms::util::Graph<NodeName, AttList>;
using TinyDomWalker = cms::util::GraphWalker<NodeName, AttList>;

void TinyDomPrettyPrint(std::ostream &, const TinyDom &);

#endif
