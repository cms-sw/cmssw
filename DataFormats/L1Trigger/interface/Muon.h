#ifndef DataFormats_L1Trigger_Muon_h
#define DataFormats_L1Trigger_Muon_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {

  class Muon;
  typedef BXVector<Muon> MuonBxCollection;
  typedef edm::Ref< MuonBxCollection > MuonRef ;
  typedef edm::RefVector< MuonBxCollection > MuonRefVector ;
  typedef std::vector< MuonRef > MuonVectorRef ;

  class Muon : public L1Candidate {
    
  public:
    Muon();

    Muon( const LorentzVector& p4,
      int pt=0,
      int eta=0,
      int phi=0,
      int qual=0,
      int charge=0,
      int chargeValid=0,
      int iso=0,
      int tfMuonIndex=-1,
      int tag=0, 
      bool debug = false,
      int isoSum = 0,
      int dPhi = 0,
      int dEta = 0,
      int rank = 0,
      int hwEtaAtVtx = 0,
      int hwPhiAtVtx = 0,
      double etaAtVtx = 0.,
      double phiAtVtx = 0.);
    
    Muon( const PolarLorentzVector& p4,
      int pt=0,
      int eta=0,
      int phi=0,
      int qual=0,
      int charge=0,
      int chargeValid=0,
      int iso=0,
      int tfMuonIndex=-1,
      int tag=0, 
      bool debug = false,
      int isoSum = 0,
      int dPhi = 0,
      int dEta = 0,
      int rank = 0,
      int hwEtaAtVtx = 0,
      int hwPhiAtVtx = 0,
      double etaAtVtx = 0.,
      double phiAtVtx = 0.);

    ~Muon();    

    // set values
    void setHwCharge(int charge);
    void setHwChargeValid(int valid);
    void setTfMuonIndex(int index);
    void setHwTag(int tag);
    
    void setHwEtaAtVtx(int hwEtaAtVtx);
    void setHwPhiAtVtx(int hwPhiAtVtx);
    void setEtaAtVtx(double etaAtVtx);
    void setPhiAtVtx(double phiAtVtx);

    void setHwIsoSum(int isoSum);
    void setHwDPhiExtra(int dPhi);
    void setHwDEtaExtra(int dEta);
    void setHwRank(int rank);

    void setDebug(bool debug);

    // methods to retrieve values
    int hwCharge() const;
    int hwChargeValid() const;
    int tfMuonIndex() const;
    int hwTag() const;

    int hwEtaAtVtx() const;
    int hwPhiAtVtx() const;
    double etaAtVtx() const;
    double phiAtVtx() const;

    int hwIsoSum() const;
    int hwDPhiExtra() const;
    int hwDEtaExtra() const;
    int hwRank() const;

    bool debug() const;
    
  private:
    
    // additional hardware quantities common to L1 global jet
    int hwCharge_;
    int hwChargeValid_;
    int tfMuonIndex_;
    int hwTag_;

    // additional hardware quantities only available if debug flag is set
    bool debug_;
    int hwIsoSum_;
    int hwDPhiExtra_;
    int hwDEtaExtra_;
    int hwRank_;
    
    // muon coordinates at the vertex
    int hwEtaAtVtx_;
    int hwPhiAtVtx_;
    double etaAtVtx_;
    double phiAtVtx_;
  };
  
}

#endif
