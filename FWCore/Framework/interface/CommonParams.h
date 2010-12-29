#ifndef FWCore_Framework_CommonParams_h
#define FWCore_Framework_CommonParams_h

/*----------------------------------------------------------------------

Class to hold parameters used by the EventProcessor and also by subprocesses.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"

#include <string>

namespace edm {
  //------------------------------------------------------------------
  //

  struct CommonParams {
    CommonParams() :
      processName_(),
      releaseVersion_(),
      passID_(),
      maxEventsInput_(),
      maxLumisInput_() {
    }

    CommonParams(std::string const& processName,
                 ReleaseVersion const& releaseVersion,
                 PassID const& passID,
                 int maxEvents,
                 int maxLumis) :
      processName_(processName),
      releaseVersion_(releaseVersion),
      passID_(passID),
      maxEventsInput_(maxEvents),
      maxLumisInput_(maxLumis) {
    }
      
    std::string processName_;
    ReleaseVersion releaseVersion_;
    PassID passID_;
    int maxEventsInput_;
    int maxLumisInput_;
  }; // struct CommonParams
}

#endif
