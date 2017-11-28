#ifndef FWCore_Framework_defaultCmsRunServices_h
#define FWCore_Framework_defaultCmsRunServices_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     defaultCmsRunServices
//
/**\class defaultCmsRunServices defaultCmsRunServices.h FWCore/Framework/interface/defaultCmsRunServices.h

 Description: Returns the names of the standard cmsRun default services

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Jan 30 13:40:06 CDT 2017
//

// user include files

// system include files

#include <vector>
#include <string>

// forward declarations

namespace edm {
  std::vector<std::string> defaultCmsRunServices();
}

#endif
