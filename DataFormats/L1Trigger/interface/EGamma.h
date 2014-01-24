#ifndef DataFormats_L1Trigger_EGamma_h
#define DataFormats_L1Trigger_EGamma_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"


namespace l1t {

  class EGamma;
  typedef BXVector<EGamma> EGammaBxCollection;

  class EGamma : public L1Candidate {
    
  public:
    EGamma(){}
    EGamma( const LorentzVector& p4,
	    int pt=0,
	    int eta=0,
	    int phi=0,
	    int qual=0,
	    int iso=0);
    
    ~EGamma();		

    
  private:
    
    // additional hardware quantities common to L1 global jet
    // there are currently none
    

    
  };
  
}

#endif
