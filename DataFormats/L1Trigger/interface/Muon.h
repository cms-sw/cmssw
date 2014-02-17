#ifndef DataFormats_L1Trigger_Muon_h
#define DataFormats_L1Trigger_Muon_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class Muon;
  typedef BXVector<Muon> MuonBxCollection;

  class Muon : public L1Candidate {
    
  public:
    Muon(){}
    Muon( const LorentzVector& p4,
	  int pt=0,
	  int eta=0,
	  int phi=0,
	  int qual=0,
	  int charge=0,
	  int chargeValid=0,
	  int iso=0,
	  int mip=0,
	  int tag=0 );
    
    ~Muon();		

    // set integer values
    void setHwCharge(int charge);
    void setHwChargeValid(int valid);
    void setHwMip(int mip);
    void setHwTag(int tag);

    // methods to retrieve integer values
    int hwCharge() const;
    int hwChargeValid() const;
    int hwMip() const;
    int hwTag() const;
    
  private:
    
    // additional hardware quantities common to L1 global jet
    int hwCharge_;
    int hwChargeValid_;
    int hwMip_;
    int hwTag_;
    
  };
  
}

#endif
