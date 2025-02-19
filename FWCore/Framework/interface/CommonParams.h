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
      maxLumisInput_() {
    }

    CommonParams(int maxEvents,
                 int maxLumis) :
      maxEventsInput_(maxEvents),
      maxLumisInput_(maxLumis) {
    }
      
    int maxEventsInput_;
    int maxLumisInput_;
  }; // struct CommonParams
}

#endif
