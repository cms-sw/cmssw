#ifndef FWCore_Framework_getAllTriggerNames_h
#define FWCore_Framework_getAllTriggerNames_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     getAllTriggerNames
// 
/**\function getAllTriggerNames getAllTriggerNames.h "FWCore/Framework/interface/getAllTriggerNames.h"

 Description: Returns a list of all the trigger names in the current process
 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 26 Jul 2013 20:40:50 GMT
// $Id$
//

// system include files
#include <vector>
#include <string>

// user include files

// forward declarations
namespace edm {
  std::vector<std::string> const& getAllTriggerNames();
}
#endif
