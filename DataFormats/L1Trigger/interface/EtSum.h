#ifndef DataFormats_L1Trigger_ETSum_h
#define DataFormats_L1Trigger_ETSum_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class EtSum;
  typedef BXVector<EtSum> EtSumBxCollection;
  typedef edm::Ref< EtSumBxCollection > EtSumRef ;
  typedef edm::RefVector< EtSumBxCollection > EtSumRefVector ;
  typedef std::vector< EtSumRef > EtSumVectorRef ;

  class EtSum : public L1Candidate {

  public:

    enum EtSumType {
      kTotalEt,
      kTotalHt,
      kMissingEt,
      kMissingHt,
      kTotalEtx,
      kTotalEty,
      kTotalHtx,
      kTotalHty,
      kMissingEt2,
      kTotalEtx2,
      kTotalEty2,
      kMinBiasHFP0,
      kMinBiasHFM0,
      kMinBiasHFP1,
      kMinBiasHFM1      
    };

    EtSum(){}
    EtSum( const LorentzVector& p4,
	   EtSumType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);

    EtSum( const PolarLorentzVector& p4,
	   EtSumType type,
	   int pt=0,
	   int eta=0,
	   int phi=0,
	   int qual=0);


    ~EtSum();

    void setType(EtSumType type);

    EtSumType getType() const;

  private:

    // type of EtSum
    EtSumType type_;

    // additional hardware quantities common to L1 global EtSum
    // there are currently none

  };

}

#endif
