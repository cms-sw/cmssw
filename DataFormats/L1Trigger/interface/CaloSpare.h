#ifndef DataFormats_L1Trigger_CaloSpare_h
#define DataFormats_L1Trigger_CaloSpare_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class CaloSpare;
  typedef BXVector<CaloSpare> CaloSpareBxCollection;

  class CaloSpare : public L1Candidate {

  public:

    enum CaloSpareType {
      HFBitCount,
      HFRingSum,
      Tau,
      Centrality,
      V2
    };

    CaloSpare(){}
    CaloSpare( const LorentzVector& p4,
	   CaloSpareType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);

    CaloSpare( const PolarLorentzVector& p4,
	   CaloSpareType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);


    ~CaloSpare();

    void setType(CaloSpareType type);

    int GetRing(unsigned index) const;
    void SetRing(unsigned index, int value);

    CaloSpareType getType() const;

  private:

    // type of CaloSpare
    CaloSpareType type_;

    // additional hardware quantities common to L1 global CaloSpare
    // there are currently none

  };

}

#endif
