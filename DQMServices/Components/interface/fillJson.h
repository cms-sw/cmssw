#ifndef DQMServices_Components_fillJson_h
#define DQMServices_Components_fillJson_h
// -*- C++ -*-
//
// Package:     DQMServices/Components
// Class  :     fillJson
//
/**\function fillJson fillJson.h "DQMServices/Components/interface/fillJson.h"

 Description: Function used by DQMFileSaver and JsonWritingTimedPoolOutputModule

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 08 Nov 2018 21:17:22 GMT
//

// system include files
#include <boost/property_tree/ptree.hpp>

// user include files

// forward declarations
namespace evf {
  class FastMonitoringService;
}

namespace dqmfilesaver {
  // used by the JsonWritingTimedPoolOutputModule,
  // fms will be nullptr in such case
  boost::property_tree::ptree fillJson(int run,
                                       int lumi,
                                       const std::string& dataFilePathName,
                                       const std::string& transferDestinationStr,
                                       const std::string& mergeTypeStr,
                                       evf::FastMonitoringService* fms);
}  // namespace dqmfilesaver

#endif
