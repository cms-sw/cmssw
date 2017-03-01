#ifndef FWCore_Framework_CommonParams_h
#define FWCore_Framework_CommonParams_h

/*----------------------------------------------------------------------

Class to hold parameters used by the EventProcessor and also by subprocesses.

----------------------------------------------------------------------*/

namespace edm {
  //------------------------------------------------------------------
  //

  struct CommonParams {
    CommonParams() :
      maxEventsInput_(),
      maxLumisInput_(),
      maxSecondsUntilRampdown_() {
    }

    CommonParams(int maxEvents,
                 int maxLumis,
                 int maxSecondsUntilRampdown) :
      maxEventsInput_(maxEvents),
      maxLumisInput_(maxLumis),
      maxSecondsUntilRampdown_(maxSecondsUntilRampdown) {
    }
      
    int maxEventsInput_;
    int maxLumisInput_;
    int maxSecondsUntilRampdown_;
  }; // struct CommonParams
}

#endif
