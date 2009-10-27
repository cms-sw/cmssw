#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentTrackSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentTrackSelector_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include <vector>

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
namespace edm {
  class Event;
  class ParameterSet;
}

class TrackingRecHit;
class SiStripRecHit2D;

class AlignmentTrackSelector
{

 public:

  typedef std::vector<const reco::Track*> Tracks; 

  /// constructor
  AlignmentTrackSelector(const edm::ParameterSet & cfg);

  /// destructor
  ~AlignmentTrackSelector();

  /// select tracks
  Tracks select(const Tracks& tracks, const edm::Event& evt) const;
  ///returns if any of the Filters is used.
  bool useThisFilter();


 private:

  /// apply basic cuts on pt,eta,phi,nhit
  Tracks basicCuts(const Tracks& tracks, const edm::Event& evt) const;
  /// checking hit requirements beyond simple number of valid hits
  bool detailedHitsCheck(const reco::Track* track, const edm::Event& evt) const;
  bool isHit2D(const TrackingRecHit &hit) const;
  /// if valid, check for minimum charge (currently only in strip), if invalid give true 
  bool isOkCharge(const TrackingRecHit* therechit) const;
  bool isOkChargeStripHit(const SiStripRecHit1D *siStripRecHit1D) const;
  bool isOkChargeStripHit(const SiStripRecHit2D *siStripRecHit2D) const;
  bool isIsolated(const TrackingRecHit* therechit, const edm::Event& evt) const;
  bool isOkTrkQuality(const reco::Track* track) const;

  /// filter the n highest pt tracks
  Tracks theNHighestPtTracks(const Tracks& tracks) const;

  //filter tracks that do not have a min # of hits taken by the Skim&Prescale workflow
  Tracks checkPrescaledHits(const Tracks& tracks, const edm::Event& evt) const;

  /// compare two tracks in pt (used by theNHighestPtTracks)
  struct ComparePt {
    bool operator()( const reco::Track* t1, const reco::Track* t2 ) const {
      return t1->pt()> t2->pt();
    }
  };
  ComparePt ptComparator;

  const bool applyBasicCuts_, applyNHighestPt_, applyMultiplicityFilter_;
  const bool seedOnlyFromAbove_, applyIsolation_, chargeCheck_ ;
  const int nHighestPt_, minMultiplicity_, maxMultiplicity_;
  const bool multiplicityOnInput_; /// if true, cut min/maxMultiplicity on input instead of on final result
  const double ptMin_,ptMax_,pMin_,pMax_,etaMin_,etaMax_,phiMin_,phiMax_,nHitMin_,nHitMax_,chi2nMax_;
  const double minHitChargeStrip_, minHitIsolation_;
  const edm::InputTag rphirecHitsTag_;
  const edm::InputTag matchedrecHitsTag_;
  const bool countStereoHitAs2D_; // count hits on stereo components of GluedDet for nHitMin2D_?
  const unsigned int nHitMin2D_;
  const int minHitsinTIB_, minHitsinTOB_, minHitsinTID_, minHitsinTEC_, minHitsinBPIX_, minHitsinFPIX_, minHitsinPIX_;

  const edm::InputTag clusterValueMapTag_;  // ValueMap containing association cluster - flag
  const int minPrescaledHits_;
  const bool applyPrescaledHitsFilter_;

  std::vector<reco::TrackBase::TrackQuality> trkQualities_;
};

#endif

