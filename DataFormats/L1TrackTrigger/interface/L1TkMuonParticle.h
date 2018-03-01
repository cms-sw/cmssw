#ifndef L1TkTrigger_L1MuonParticle_h
#define L1TkTrigger_L1MuonParticle_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

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

      //! more basic constructor, in case refs/ptrs can't be set or to be set separately
      L1TkMuonParticle(const L1Candidate& cand) : L1Candidate(cand), theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

      virtual ~L1TkMuonParticle() {}


      const edm::Ptr< L1TTTrackType >& getTrkPtr() const
      { return trkPtr_ ; }

      const edm::Ref< l1t::RegionalMuonCandBxCollection >& getMuRef() const
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
      edm::Ref< l1t::RegionalMuonCandBxCollection > muRef_ ;

      edm::Ptr< L1TTTrackType > trkPtr_ ;

      float theIsolation;
      float TrkzVtx_ ;
      unsigned int quality_;
  };
}

#endif

