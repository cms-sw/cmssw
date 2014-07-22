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
      maxGracefulRuntime_() {
    }

    CommonParams(int maxEvents,
                 int maxLumis,
                 int maxGracefulRuntime) :
      maxEventsInput_(maxEvents),
      maxLumisInput_(maxLumis),
      maxGracefulRuntime_(maxGracefulRuntime) {
    }
      
    int maxEventsInput_;
    int maxLumisInput_;
    int maxGracefulRuntime_;
  }; // struct CommonParams
}

#endif
