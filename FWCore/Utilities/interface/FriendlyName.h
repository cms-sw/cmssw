#ifndef FWCore_Utilities_FriendlyName_h
#define FWCore_Utilities_FriendlyName_h
/*
 *  friendlyName.h
 *  CMSSW
 *
 *  Created by Chris Jones on 2/24/06.
 *
 */
#include <string>

namespace edm {
  namespace friendlyname {
    std::string friendlyName(std::string const& iFullName);
  }
}
#endif
