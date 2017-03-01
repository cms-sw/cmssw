// -*- ++ -*-
//
// Package:     FWCore/Framework
// Function:    defaultCmsRunServices
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Tue, 06 Sep 2016 16:04:28 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/defaultCmsRunServices.h"

namespace edm {
   std::vector<std::string> defaultCmsRunServices() {
      std::vector<std::string> returnValue = {"MessageLogger",
                                              "InitRootHandlers",
#ifdef linux
                                              "EnableFloatingPointExceptions",
#endif
                                              "UnixSignalService",
                                              "AdaptorConfig",
                                              "SiteLocalConfigService",
                                              "StatisticsSenderService",
                                              "CondorStatusService"};

      return returnValue;
   }
   
}