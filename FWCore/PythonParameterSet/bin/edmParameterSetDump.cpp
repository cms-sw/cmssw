// -*- C++ -*-
//
// Package:     PythonParameterSet
// Class  :     edmParameterSetDump
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Rick Wilkinson
//         Created:  Thu Aug  2 13:33:53 EDT 2007
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "boost/shared_ptr.hpp"
#include <iostream>
#include <string>

int main (int argc, char **argv) {
  if(argc != 2) {
    std::cout << "Usage: edmParameterSetDump <cfgfile>" << std::endl;
  }
  std::string fileName(argv[1]);
  boost::shared_ptr<edm::ParameterSet> parameterSet = edm::readConfig(fileName);
  std::cout << "====Main Process====" << std::endl;
  std::cout << parameterSet->dump() << std::endl;
}
