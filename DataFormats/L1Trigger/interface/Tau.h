#ifndef DataFormats_L1Trigger_Tau_h
#define DataFormats_L1Trigger_Tau_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class Tau;
  typedef BXVector<Tau> TauBxCollection;
  typedef edm::Ref< TauBxCollection > TauRef ;
  typedef edm::RefVector< TauBxCollection > TauRefVector ;
  typedef std::vector< TauRef > TauVectorRef ;

  class Tau : public L1Candidate {

  public:
    Tau(){}
    Tau( const LorentzVector& p4,
	    int pt=0,
	    int eta=0,
	    int phi=0,
	    int qual=0,
	    int iso=0);
    Tau( const PolarLorentzVector& p4,
	    int pt=0,
	    int eta=0,
	    int phi=0,
	    int qual=0,
	    int iso=0);


    ~Tau();

  private:

    // additional hardware quantities common to L1 global jet
    // there are currently none


  };

}

#endif
