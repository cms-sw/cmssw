#ifndef DataFormats_L1Trigger_CaloEmCand_h
#define DataFormats_L1Trigger_CaloEmCand_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"

namespace l1t {

  class CaloEmCand : public L1Candidate {

  public:
    CaloEmCand(){}
    CaloEmCand( const LorentzVector& p4,
		int pt=0,
		int eta=0,
		int phi=0,
		int qual=0
		);

    ~CaloEmCand() override;

  private:
    //

  };

  typedef BXVector<CaloEmCand> CaloEmCandBxCollection;

}

#endif
