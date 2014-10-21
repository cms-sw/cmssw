#ifndef DataFormats_L1Trigger_CaloStage1Cluster_h
#define DataFormats_L1Trigger_CaloStage1Cluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  
  class CaloStage1Cluster : public L1Candidate {
    
  public:
    CaloStage1Cluster(){}
    CaloStage1Cluster( const LorentzVector p4,
		       int pt=0,
		       int eta=0,
		       int phi=0,
		       int qual=0
		       );
    
    ~CaloStage1Cluster();

  private:
    //
    
  };

  typedef BXVector<CaloStage1Cluster> CaloStage1ClusterBxCollection;
  
}

#endif
