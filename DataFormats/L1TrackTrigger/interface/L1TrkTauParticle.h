#ifndef L1TkTrigger_L1TrkTauParticle_h
#define L1TkTrigger_L1TrkTauParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TrkTauParticle
// 

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t { 

  class L1TrkTauParticle ;
  
  typedef std::vector< L1TrkTauParticle > L1TrkTauParticleCollection ;
  
  typedef edm::Ref< L1TrkTauParticleCollection > L1TrkTauParticleRef ;
  typedef edm::RefVector< L1TrkTauParticleCollection > L1TrkTauParticleRefVector ;
  typedef std::vector< L1TrkTauParticleRef > L1TrkTauParticleVectorRef ;
  
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollection;
  typedef edm::Ptr< L1TTTrackType > L1TTTrackRefPtr;
  typedef std::vector< L1TTTrackRefPtr > L1TTTrackRefPtr_Collection;

  class L1TrkTauParticle : public L1Candidate
  {     
    
  public:
    
    L1TrkTauParticle();
    

    L1TrkTauParticle( const LorentzVector& p4,
		    const std::vector< L1TTTrackRefPtr >& clustTracks,
		    float vtxIso = -999. );

    virtual ~L1TrkTauParticle() {}
      
    // ---------- const member functions ---------------------

    const L1TTTrackRefPtr getSeedTrk() const { return clustTracks_.at(0); }

    const std::vector< L1TTTrackRefPtr > getTrks() const { return clustTracks_; }

    float getVtxIso() const { return vtxIso_ ; } 
    
    // ---------- member functions ---------------------------
    
    void setVtxIso(float VtxIso)  { vtxIso_ = VtxIso ; }
    
    //	 int bx() const;
    
  private:
    std::vector< L1TTTrackRefPtr > clustTracks_;
    float vtxIso_;

  };
}

#endif


