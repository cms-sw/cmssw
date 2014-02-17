#ifndef DataFormats_L1Trigger_CaloTower_h
#define DataFormats_L1Trigger_CaloTower_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  
  class CaloTower;
  typedef BXVector<CaloTower> CaloTowerBxCollection;

  class CaloTower : public L1Candidate {
    
  public:
    CaloTower(){}
    CaloTower( const LorentzVector& p4,
	       double etEm=0.,
	       double etHad=0.,
	       int pt=0,
	       int eta=0,
	       int phi=0,
	       int qual=0,
	       int hwEtEm=0,
	       int hwEtHad=0,
	       int hwFGEm=0);
    
    ~CaloTower();

    void setEtEm( double et );
    void setEtHad( double et );
    void setHwEtEm( int et );
    void setHwFGEm( int fg );
    void setHwEtHad( int et );

    double etEm();
    double etHad();
    int hwEtEm();
    int hwFGEm();
    int hwEtHad();

  private:
    
    // additional hardware quantities
    double etEm_;
    double etHad_;
    
    int hwEtEm_;
    int hwFGEm_;

    int hwEtHad_;
    
    
  };
  
}

#endif
