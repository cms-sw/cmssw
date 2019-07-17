#ifndef L1TkTrigger_L1MuonParticle_h
#define L1TkTrigger_L1MuonParticle_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"

namespace l1t
{
  class L1TkMuonParticle : public L1Candidate
  {
    public:

    typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
    typedef std::vector< L1TTTrackType > L1TTTrackCollection;

      L1TkMuonParticle() : theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

      L1TkMuonParticle( const LorentzVector& p4,
   		        const edm::Ref< l1t::RegionalMuonCandBxCollection >& muRef,
		        const edm::Ptr< L1TTTrackType >& trkPtr,
		        float tkisol = -999. );

      //One more constructor for Tracker+ Stubs algorithm not requiring the Muon candidate
      L1TkMuonParticle( const LorentzVector& p4,
		        const edm::Ptr< L1TTTrackType >& trkPtr,
		        float tkisol = -999. );

      
      //! more basic constructor, in case refs/ptrs can't be set or to be set separately
      L1TkMuonParticle(const L1Candidate& cand) : L1Candidate(cand), theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

      virtual ~L1TkMuonParticle() {}


      const edm::Ptr< L1TTTrackType >& getTrkPtr() const
      { return trkPtr_ ; }

      const edm::Ref< l1t::RegionalMuonCandBxCollection >& getMuRef() const
      { return muRef_ ; }

      float getTrkIsol() const { return theIsolation; }
      float getTrkzVtx() const { return TrkzVtx_ ; }

      const L1MuKBMTCombinedStubRefVector getBarrelStubs() const { return barrelStubs_; }
      float dR()  const { return dR_;}
      int nTracksMatched() const { return nTracksMatch_;}
      double trackCurvature()  const { return trackCurvature_;}

      unsigned int quality()  const {return quality_;}

      unsigned int muonDetector() const {return muonDetector_;}  

      void setTrkPtr(const edm::Ptr< L1TTTrackType >& p) {trkPtr_ = p;}

      void setTrkzVtx(float TrkzVtx) { TrkzVtx_ = TrkzVtx ; }
      void setTrkIsol(float TrkIsol) { theIsolation = TrkIsol ; }
      void setQuality(unsigned int q){ quality_ = q;}
      void addBarrelStub(const L1MuKBMTCombinedStubRef&);

      void setdR(float dR) { dR_=dR;}
      void setNTracksMatched(int nTracksMatch) { nTracksMatch_=nTracksMatch;}
      void setTrackCurvature(double trackCurvature) { trackCurvature_=trackCurvature;} // this is signed

      void setMuonDetector(unsigned int detector) {muonDetector_=detector;}

    private:


	// used for the Naive producer
      edm::Ref< l1t::RegionalMuonCandBxCollection > muRef_ ;

      //Used for Tracker + Stubs algorithm
      L1MuKBMTCombinedStubRefVector barrelStubs_;

      edm::Ptr< L1TTTrackType > trkPtr_ ;

      float theIsolation;
      float TrkzVtx_ ;
      float dR_;
      int nTracksMatch_;
      double trackCurvature_;

      unsigned int quality_;
 
      int muonDetector_;     

  };
}

#endif

