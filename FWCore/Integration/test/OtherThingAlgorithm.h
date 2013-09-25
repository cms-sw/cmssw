#ifndef Integration_OtherThingAlgorithm_h
#define Integration_OtherThingAlgorithm_h

#include <string>
#include "DataFormats/TestObjects/interface/OtherThingCollectionfwd.h"
#include "DataFormats/TestObjects/interface/ThingCollectionfwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edmtest {

  class OtherThingAlgorithm {
  public:
    OtherThingAlgorithm() : theDebugLevel(0) {}
  
    /// Runs the algorithm and returns a list of OtherThings
    /// The user declares the vector and calls this method.
    void run(edm::Handle<ThingCollection> const& iThingHandle,
             OtherThingCollection& otherThingCollection,
             bool useRefs = true,
             bool refsAreTransient = false);
  
  private:
    int    theDebugLevel;
  };

}
  
#endif
