#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentTrackSelector_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentTrackSelector_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <vector>

namespace edm { class Event; }

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

  /// filter the n highest pt tracks
  Tracks theNHighestPtTracks(const Tracks& tracks) const;

  /// compare two tracks in pt (used by theNHighestPtTracks)
  struct ComparePt {
    bool operator()( const reco::Track* t1, const reco::Track* t2 ) const {
      return t1->pt()> t2->pt();
    }
  };
  ComparePt ptComparator;

  /// private data members
  edm::ParameterSet conf_;

  bool applyBasicCuts,applyNHighestPt,applyMultiplicityFilter;
  int nHighestPt,minMultiplicity,maxMultiplicity;
  double ptMin,ptMax,etaMin,etaMax,phiMin,phiMax,nHitMin,nHitMax,chi2nMax;
  int minHitsinTIB, minHitsinTOB, minHitsinTID, minHitsinTEC;

  TrackerAlignableId *TkMap;
};

#endif

