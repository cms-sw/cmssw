#ifndef L1TkTrigger_L1ElectronParticle_h
#define L1TkTrigger_L1ElectronParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


         
namespace l1t {
         
  class L1TkElectronParticle : public L1TkEmParticle 
  {
    
  public:
    
    typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
    typedef std::vector< L1TTTrackType > L1TTTrackCollection;
    
    
    L1TkElectronParticle();
    
    L1TkElectronParticle( const LorentzVector& p4,
			  const edm::Ref< EGammaBxCollection > & egRef,
			  const edm::Ptr< L1TTTrackType >& trkPtr,
			  float tkisol = -999. );
    
    virtual ~L1TkElectronParticle() {}
    
    // ---------- const member functions ---------------------
    
    const edm::Ptr< L1TTTrackType >& getTrkPtr() const
    { return trkPtr_ ; }
    
    float getTrkzVtx() const { return TrkzVtx_ ; }
    
    
    // ---------- member functions ---------------------------
    
    void setTrkzVtx(float TrkzVtx)  { TrkzVtx_ = TrkzVtx ; }
    
  private:
    
    edm::Ptr< L1TTTrackType > trkPtr_ ;
    float TrkzVtx_ ;
    
  };
}
#endif


