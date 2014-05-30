#ifndef L1Trigger_L1MuonParticleExtended_h
#define L1Trigger_L1MuonParticleExtended_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1MuonParticleExtended
// 
// Description: L1MuonParticle with detector layer information.
//              Should be useful from the start for detailed analysis of efficiency etc
//              Expect this to become feasible for future upgrade hardware.

// user include files
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/DetId/interface/DetId.h"

// forward declarations

namespace l1extra {
  
  class L1MuonParticleExtended : public L1MuonParticle  {
  public:
    L1MuonParticleExtended() : sigmaEta_(-99), sigmaPhi_(-99), quality_(0) {}
      
    L1MuonParticleExtended( const L1MuonParticle& l1mu) : L1MuonParticle(l1mu), sigmaEta_(-99), sigmaPhi_(-99), quality_(0) {}
	
    virtual ~L1MuonParticleExtended() {}
	

    struct StationData {
      StationData() : station(-1), ringOrWheel(-99), bx(-99), quality(-99), 
		      phi(-99), sigmaPhi(-99), 
		      eta(-99), sigmaEta(-99),
		      bendPhi(-99), bendEta(-99), 
		      bendPhiInt(-99), bendEtaInt(-99), 
		      valid(false) {}
      DetId id;
      int station; //a bit wasteful
      int ringOrWheel;
      int bx;
      int quality;
      float phi;
      float sigmaPhi;
      float eta;
      float sigmaEta;
      float bendPhi; //! direction.phi - position.phi
      float bendEta; //! direction.eta - position.eta 

      int bendPhiInt; //magic word from primitives about phi direction
      int bendEtaInt; //magic word from primitives about eta or theta direction

      bool valid;
    };

    // ---------- const member functions ---------------------
    virtual L1MuonParticleExtended* clone() const
    { return new L1MuonParticleExtended( *this ) ; }
    
    // ---------- member functions ---------------------------
    const L1MuRegionalCand& cscCand() const {return cscCand_;}
    void setCscCand(const L1MuRegionalCand& cand) {cscCand_ = cand;}

    const L1MuRegionalCand& rpcCand() const {return rpcCand_;}
    void setRpcCand(const L1MuRegionalCand& cand) {rpcCand_ = cand;}

    const L1MuRegionalCand& dtCand() const {return dtCand_;}
    void setDtCand(const L1MuRegionalCand& cand) {dtCand_ = cand;}

    float sigmaEta() const { return sigmaEta_; }
    void  setSigmaEta(float val) {sigmaEta_ = val;}

    float sigmaPhi() const { return sigmaPhi_; }
    void  setSigmaPhi(float val) {sigmaPhi_ = val;}

    unsigned int quality() const {return quality_;}
    void setQuality(unsigned int q) {quality_ = q;}

    const StationData& cscData(unsigned int s ) const {return (s>0 && s<= 4) ? cscData_[s-1] : dummyData_; }
    void setCscData(const StationData& sd, unsigned int s) {if (s>0 && s<= 4) cscData_[s-1] = sd; }

    const StationData& dtData(unsigned int s ) const {return (s>0 && s<= 4) ? dtData_[s-1] : dummyData_; }
    void setDtData(const StationData& sd, unsigned int s) {if (s>0 && s<= 4) dtData_[s-1] = sd; }

    const StationData& rpcData(unsigned int s ) const {return (s>0 && s<= 4) ? rpcData_[s-1] : dummyData_; }
    void setRpcData(const StationData& sd, unsigned int s) {if (s>0 && s<= 4) rpcData_[s-1] = sd; }
  private:
    // ---------- member data --------------------------------
    L1MuRegionalCand cscCand_;
    L1MuRegionalCand rpcCand_;
    L1MuRegionalCand dtCand_;

    float sigmaEta_;
    float sigmaPhi_;

    unsigned int quality_;

    StationData cscData_[4];
    StationData dtData_[4];
    StationData rpcData_[4];

    StationData dummyData_;
  };
}

#endif
