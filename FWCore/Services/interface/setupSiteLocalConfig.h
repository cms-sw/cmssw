#ifndef FWCore_Services_setupSiteLocalConfig_h
#define FWCore_Services_setupSiteLocalConfig_h

/** \function edm::setupSiteLocalConfig

  Description: Setups up the SiteLocalConfig service.

  Usage:
*/

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

namespace edm {
  ServiceRegistry::Operate setupSiteLocalConfig();
}  // namespace edm
#endif
