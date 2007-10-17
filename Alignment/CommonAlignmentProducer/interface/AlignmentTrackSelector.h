#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentTrackSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentTrackSelector_h

#include "DataFormats/TrackReco/interface/Track.h"

#include <vector>

namespace edm {
  class Event;
  class ParameterSet;
}

class TrackingRecHit;

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

 private:

  /// apply basic cuts on pt,eta,phi,nhit
  Tracks basicCuts(const Tracks& tracks) const;
  /// checking hit requirements beyond simple number of valid hits
  bool detailedHitsCheck(const reco::Track* track) const;
  bool isHit2D(const TrackingRecHit &hit) const;


  /// filter the n highest pt tracks
  Tracks theNHighestPtTracks(const Tracks& tracks) const;

  /// compare two tracks in pt (used by theNHighestPtTracks)
  struct ComparePt {
    bool operator()( const reco::Track* t1, const reco::Track* t2 ) const {
      return t1->pt()> t2->pt();
    }
  };
  ComparePt ptComparator;

  const bool applyBasicCuts,applyNHighestPt,applyMultiplicityFilter;
  const int nHighestPt,minMultiplicity,maxMultiplicity;
  const bool multiplicityOnInput; /// if true, cut min/maxMultiplicity on input instead of on final result
  const double ptMin,ptMax,etaMin,etaMax,phiMin,phiMax,nHitMin,nHitMax,chi2nMax;
  const unsigned int nHitMin2D;
  int minHitsinTIB, minHitsinTOB, minHitsinTID, minHitsinTEC, minHitsinBPIX, minHitsinFPIX;

};

#endif

