// include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

// This HLTFilter will accept only events with HCAL zero suppression enabled.
// The condition is checked looking at the event number: 
//   - by construction, events with the lowest 12 bits of their L1A number (mirrored in the framework's event number)
//     equal to 0 will be read *without* Zero Suppression (NZS mode).
//   - all other events will have Zero Suppression (ZS mode).

class HLTHcalZSFilter : public HLTFilter {
public:
  explicit HLTHcalZSFilter(const edm::ParameterSet & config)
  { }
  
  ~HLTHcalZSFilter(void) 
  { }

private:
  virtual bool filter(edm::Event & event, const edm::EventSetup & setup) 
  {
    // return true for all events with the lowest 12 bits different from 0x.....000
    return bool(event.id().event() & 0x00000FFF);
  }
};

// define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTHcalZSFilter);
