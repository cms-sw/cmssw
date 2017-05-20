#ifndef L1TkTrigger_L1MuonParticle_h
#define L1TkTrigger_L1MuonParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkMuonParticle

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"


#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t
{
  class L1TkMuonParticle : public L1Candidate
  {
    public:

    typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
    typedef std::vector< L1TTTrackType > L1TTTrackCollection;

      L1TkMuonParticle() : theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

      L1TkMuonParticle( const LorentzVector& p4,
   		        const edm::Ref< MuonBxCollection >& muRef,
		        const edm::Ptr< L1TTTrackType >& trkPtr,
		        float tkisol = -999. );

      //! more basic constructor, in case refs/ptrs can't be set or to be set separately
      L1TkMuonParticle(const L1Candidate& cand) : L1Candidate(cand), theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

      virtual ~L1TkMuonParticle() {}


      const edm::Ptr< L1TTTrackType >& getTrkPtr() const
         { return trkPtr_ ; }

      const edm::Ref< MuonBxCollection >& getMuRef() const
	{ return muRef_ ; }
    
      float getTrkIsol() const { return theIsolation; }
      float getTrkzVtx() const { return TrkzVtx_ ; }


      unsigned int quality()  const {return quality_;}

      void setTrkPtr(const edm::Ptr< L1TTTrackType >& p) {trkPtr_ = p;}
      
      void setTrkzVtx(float TrkzVtx) { TrkzVtx_ = TrkzVtx ; }
      void setTrkIsol(float TrkIsol) { theIsolation = TrkIsol ; }

      void setQuality(unsigned int q){ quality_ = q;}

    private:


	// used for the Naive producer
      edm::Ref< MuonBxCollection > muRef_ ;

      edm::Ptr< L1TTTrackType > trkPtr_ ;

      float theIsolation;
      float TrkzVtx_ ;
      unsigned int quality_;
  };
}

#endif

