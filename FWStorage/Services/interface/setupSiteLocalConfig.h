#ifndef FWCore_Services_setupSiteLocalConfig_h
#define FWCore_Services_setupSiteLocalConfig_h

/** \function edm::setupSiteLocalConfig

  Description: Setups up the SiteLocalConfig service.

  Usage:
*/

#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

namespace edm {
  /// Makes the SiteLocalConfig service available, and keeps it alive, as long as the returned object exists.
  ServiceRegistry::LockedOperate setupSiteLocalConfig();
}  // namespace edm
#endif
