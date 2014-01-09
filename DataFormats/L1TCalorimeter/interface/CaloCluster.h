#ifndef DataFormats_L1Trigger_CaloCluster_h
#define DataFormats_L1Trigger_CaloCluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  
  class CaloCluster : public L1Candidate {
    
  public:
    CaloCluster(){}
    CaloCluster( const LorentzVector p4,
		       int pt=0,
		       int eta=0,
		       int phi=0,
		       int qual=0
		       );
    
    ~CaloCluster();

  private:
    //
    
  };

  typedef BXVector<CaloCluster> CaloClusterBxCollection;
  
}

#endif
