#ifndef x_TinyDom2_h
#define x_TinyDom2_h

#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include "DetectorDescription/RegressionTest/src/TagName.h"

class AnotherDummy2 {};

  typedef std::map<TagName, TagName> AttList2;
  typedef std::pair<TagName, AttList2> Node2;
  typedef graph<Node2, AnotherDummy2> TinyDom2;
  typedef graphwalker<Node2, AnotherDummy2> TinyDom2Walker;
  
  void TinyDom2PrettyPrint(std::ostream &, const TinyDom2 &);

#endif
