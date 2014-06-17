#ifndef DataFormats_L1Trigger_HFRingSum_h
#define DataFormats_L1Trigger_HFRingSum_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class HFRingSum;
  typedef BXVector<HFRingSum> HFRingSumBxCollection;

  class HFRingSum : public L1Candidate {

  public:

    enum HFRingSumType {
      RealHFRingSum,
      Tau,
      Centrality,
      V2
    };

    HFRingSum(){}
    HFRingSum( const LorentzVector& p4,
	   HFRingSumType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);

    HFRingSum( const PolarLorentzVector& p4,
	   HFRingSumType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);


    ~HFRingSum();

    void setType(HFRingSumType type);

    HFRingSumType getType() const;

  private:

    // type of HFRingSum
    HFRingSumType type_;

    // additional hardware quantities common to L1 global HFRingSum
    // there are currently none

  };

}

#endif
