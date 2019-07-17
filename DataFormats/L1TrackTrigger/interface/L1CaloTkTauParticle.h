#ifndef L1TkTrigger_L1CaloTkTauParticle_h
#define L1TkTrigger_L1CaloTkTauParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1CaloTkTauParticle
// 

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t { 

  class L1CaloTkTauParticle ;
  
  typedef std::vector< L1CaloTkTauParticle > L1CaloTkTauParticleCollection ;
  
  typedef edm::Ref< L1CaloTkTauParticleCollection > L1CaloTkTauParticleRef ;
  typedef edm::RefVector< L1CaloTkTauParticleCollection > L1CaloTkTauParticleRefVector ;
  typedef std::vector< L1CaloTkTauParticleRef > L1CaloTkTauParticleVectorRef ;
  
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollection;
  typedef edm::Ptr< L1TTTrackType > L1TTTrackRefPtr;
  typedef std::vector< L1TTTrackRefPtr > L1TTTrackRefPtr_Collection;

  class L1CaloTkTauParticle : public L1Candidate
  {     
    
  public:
    
    L1CaloTkTauParticle();

    L1CaloTkTauParticle( const LorentzVector& p4, // caloTau calibrated p4
			 const LorentzVector& tracksP4,
			 const std::vector< L1TTTrackRefPtr >& clustTracks,
			 Tau& caloTau,
			 float vtxIso = -999.); 
			 //float Et = -999. ); // calibrated Et

    virtual ~L1CaloTkTauParticle() {}
    
    // ---------- const member functions ---------------------

    const L1TTTrackRefPtr getSeedTrk() const { return clustTracks_.at(0); }

    float getVtxIso() const { return vtxIso_ ; } 
    
    LorentzVector getTrackBasedP4() const { return tracksP4_ ; } 

    float getTrackBasedEt() const { return tracksP4_.Et() ; } 
    
    Tau getCaloTau() const { return caloTau_; }    
    
    // ---------- member functions ---------------------------
    
    void setVtxIso(float VtxIso)  { vtxIso_ = VtxIso ; }

    //void setEt(float Et)  { Et_ = Et ; }
        
  private:
    LorentzVector tracksP4_;
    std::vector< L1TTTrackRefPtr > clustTracks_;
    Tau caloTau_;
    float vtxIso_;
    //float Et_; // calibrated Et

  };
}

#endif


