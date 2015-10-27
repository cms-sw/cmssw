#ifndef DataFormats_L1Trigger_Muon_h
#define DataFormats_L1Trigger_Muon_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class Muon;
  typedef BXVector<Muon> MuonBxCollection;

  class Muon : public L1Candidate {
    
  public:
    Muon() {};
    Muon( const LorentzVector& p4,
      int pt=0,
      int eta=0,
      int phi=0,
      int qual=0,
      int charge=0,
      int chargeValid=0,
      int iso=0,
      int tag=0, 
      bool debug = false,
      int isoSum = 0,
      int dPhi = 0,
      int dEta = 0,
      int rank = 0);
    
    Muon( const PolarLorentzVector& p4,
      int pt=0,
      int eta=0,
      int phi=0,
      int qual=0,
      int charge=0,
      int chargeValid=0,
      int iso=0,
      int tag=0, 
      bool debug = false,
      int isoSum = 0,
      int dPhi = 0,
      int dEta = 0,
      int rank = 0);

    ~Muon();    

    // set integer values
    void setHwCharge(int charge);
    void setHwChargeValid(int valid);
    void setHwTag(int tag);
    
    void setHwIsoSum(int isoSum);
    void setHwDPhiExtra(int dPhi);
    void setHwDEtaExtra(int dEta);
    void setHwRank(int rank);

    // methods to retrieve integer values
    int hwCharge() const;
    int hwChargeValid() const;
    int hwTag() const;

    int hwIsoSum() const;
    int hwDPhiExtra() const;
    int hwDEtaExtra() const;  
    int hwRank() const;  
    
    
  private:
    
    // additional hardware quantities common to L1 global jet
    int hwCharge_;
    int hwChargeValid_;
    int hwTag_;

    // additional hardware quantities only available if debug flag is set
    bool debug_;
    int hwIsoSum_;
    int hwDPhiExtra_;
    int hwDEtaExtra_;
    int hwRank_;
    
  };
  
}

#endif
