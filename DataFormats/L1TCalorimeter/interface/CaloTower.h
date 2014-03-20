#ifndef DataFormats_L1TCalorimeter_CaloTower_h
#define DataFormats_L1TCalorimeter_CaloTower_h


#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  
  class CaloTower;
  typedef BXVector<CaloTower> CaloTowerBxCollection;
  
  class CaloTower : public L1Candidate {
    
  public:
    
  CaloTower(): etEm_(0.),etHad_(0.),hwEtEm_(0),hwEtHad_(0),hwEtRatio_(0) {}
    
    CaloTower( const LorentzVector& p4,
	       double etEm=0.,
	       double etHad=0.,
	       int pt=0,
	       int eta=0,
	       int phi=0,
	       int qual=0,
	       int hwEtEm=0,
	       int hwEtHad=0,
	       int hwEtRatio=0);
    
    ~CaloTower();

    void setEtEm( double et );
    void setEtHad( double et );

    void setHwEtEm( int et );
    void setHwEtHad( int et );
    void setHwEtRatio( int ratio );

    double etEm()const;
    double etHad()const;

    int hwEtEm()const;
    int hwEtHad()const;
    int hwEtRatio()const;

  private:
    
    // additional hardware quantities
    double etEm_;
    double etHad_;
    
    int hwEtEm_;
    int hwEtHad_;
    int hwEtRatio_;
    
  };
  
}

#endif
