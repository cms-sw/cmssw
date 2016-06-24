#ifndef x_TinyDom_h
#define x_TinyDom_h

#include <iostream>
#include <map>
#include <string>

#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include "DetectorDescription/RegressionTest/src/TagName.h"

class AnotherDummy {};

  typedef TagName NodeName;
  typedef TagName AttName;
  typedef TagName AttValue;
  typedef std::map<AttName,AttValue> AttList;
  typedef graph<NodeName, AttList> TinyDom;
  typedef graphwalker<NodeName, AttList> TinyDomWalker;
  
  void TinyDomPrettyPrint(std::ostream &, const TinyDom &);

#endif
