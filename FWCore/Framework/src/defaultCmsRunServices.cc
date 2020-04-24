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
                                              "UnixSignalService",
                                              "AdaptorConfig",
                                              "SiteLocalConfigService",
                                              "StatisticsSenderService",
                                              "CondorStatusService",
                                              "XrdAdaptor::XrdStatisticsService"};

      return returnValue;
   }
   
}
