#ifndef Framework_FriendlyName_h
#define Framework_FriendlyName_h
/*
 *  friendlyName.h
 *  CMSSW
 *
 *  Created by Chris Jones on 2/24/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */
#include <string>

namespace edm {
  namespace friendlyname {
    std::string friendlyName(std::string const& iFullName);
  }
}
#endif
