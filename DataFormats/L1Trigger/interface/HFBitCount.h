#ifndef DataFormats_L1Trigger_HFBitCount_h
#define DataFormats_L1Trigger_HFBitCount_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class HFBitCount;
  typedef BXVector<HFBitCount> HFBitCountBxCollection;

  class HFBitCount : public L1Candidate {

  public:

    enum HFBitCountType {
      RealHFBitCount,
      Tau,
      Centrality,
      V2
    };

    HFBitCount(){}
    HFBitCount( const LorentzVector& p4,
	   HFBitCountType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);

    HFBitCount( const PolarLorentzVector& p4,
	   HFBitCountType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);


    ~HFBitCount();

    void setType(HFBitCountType type);

    HFBitCountType getType() const;

  private:

    // type of HFBitCount
    HFBitCountType type_;

    // additional hardware quantities common to L1 global HFBitCount
    // there are currently none

  };

}

#endif
