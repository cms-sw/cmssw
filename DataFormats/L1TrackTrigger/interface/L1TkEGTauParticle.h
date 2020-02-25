#ifndef L1TkTrigger_L1TkEGTauParticle_h
#define L1TkTrigger_L1TkEGTauParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEGTauParticle
// 

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t { 

  class L1TkEGTauParticle ;
  
  typedef std::vector< L1TkEGTauParticle > L1TkEGTauParticleCollection ;
  
  typedef edm::Ref< L1TkEGTauParticleCollection > L1TkEGTauParticleRef ;
  typedef edm::RefVector< L1TkEGTauParticleCollection > L1TkEGTauParticleRefVector ;
  typedef std::vector< L1TkEGTauParticleRef > L1TkEGTauParticleVectorRef ;
  
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollection;
  typedef edm::Ptr< L1TTTrackType > L1TTTrackRefPtr;
  typedef std::vector< L1TTTrackRefPtr > L1TTTrackRefPtr_Collection;
  typedef edm::Ref< EGammaBxCollection > EGammaRef ;
  typedef std::vector< EGammaRef > EGammaVectorRef ;

  class L1TkEGTauParticle : public L1Candidate
  {     
    
  public:
    
    L1TkEGTauParticle();
    

    L1TkEGTauParticle( const LorentzVector& p4,
		    const std::vector< L1TTTrackRefPtr >& clustTracks,
		    const std::vector< EGammaRef >& clustEGs,
		    float iso = -999. );

    virtual ~L1TkEGTauParticle() {}
      
    // ---------- const member functions ---------------------

    const L1TTTrackRefPtr getSeedTrk() const { return clustTracks_.at(0); }

    const std::vector< L1TTTrackRefPtr > getTrks() const { return clustTracks_; }
    
    const std::vector< EGammaRef > getEGs() const { return clustEGs_; }

    float getIso() const { return iso_ ; }     

    // ---------- member functions ---------------------------
    
    void setVtxIso(float iso)  { iso_ = iso ; }
    
    //	 int bx() const;
    
  private:
    std::vector< L1TTTrackRefPtr > clustTracks_;
    std::vector< EGammaRef > clustEGs_;
    float iso_;

  };
}

#endif


