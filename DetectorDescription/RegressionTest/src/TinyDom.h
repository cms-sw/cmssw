#ifndef x_TinyDom_h
#define x_TinyDom_h

#include "DetectorDescription/RegressionTest/src/TagName.h"
#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include <string>
#include <map>
#include <iostream>

class AnotherDummy {};

  typedef TagName NodeName;
  typedef TagName AttName;
  typedef TagName AttValue;
  typedef std::map<AttName,AttValue> AttList;
  typedef graph<NodeName, AttList> TinyDom;
  typedef graphwalker<NodeName, AttList> TinyDomWalker;
  
  void TinyDomPrettyPrint(std::ostream &, const TinyDom &);

#endif
