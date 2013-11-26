#ifndef DataFormats_L1Trigger_Jet_h
#define DataFormats_L1Trigger_Jet_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"

namespace l1t {

  class Jet : public L1Candidate {
    
  public:
  Jet( const LorentzVector& p4,
       int pt=0,
       int eta=0,
       int phi=0,
       int qual=0);
    
  ~Jet();		
  
  private:
  
  // additional hardware quantities common to L1 global jet
  // there are currently none
  
  };
  
}

#endif
