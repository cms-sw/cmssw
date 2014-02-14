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
    EGamma(const L1Candidate& rhs):L1Candidate(rhs){} //this is okay given that EGamma currently has no additional data members 
                                                      //but need to add a check for rhs being an EGamma if this changes

    ~EGamma();		

    
  private:
    
    // additional hardware quantities common to L1 global jet
    // there are currently none
    

    
  };
  
}

#endif
