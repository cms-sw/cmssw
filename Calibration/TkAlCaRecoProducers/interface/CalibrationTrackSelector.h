#ifndef Calibration_TkAlCaRecoProducers_CalibrationTrackSelector_h
#define Calibration_TkAlCaRecoProducers_CalibrationTrackSelector_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>

namespace edm {
  class Event;
  class ParameterSet;
}  // namespace edm

class TrackingRecHit;

class CalibrationTrackSelector {
public:
  typedef std::vector<const reco::Track *> Tracks;

  /// constructor
  CalibrationTrackSelector(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC);

  /// destructor
  ~CalibrationTrackSelector();

  /// select tracks
  Tracks select(const Tracks &tracks, const edm::Event &evt) const;

private:
  /// apply basic cuts on pt,eta,phi,nhit
  Tracks basicCuts(const Tracks &tracks, const edm::Event &evt) const;
  /// checking hit requirements beyond simple number of valid hits
  bool detailedHitsCheck(const reco::Track *track, const edm::Event &evt) const;
  bool isHit2D(const TrackingRecHit &hit) const;
  bool isOkCharge(const TrackingRecHit *therechit) const;
  bool isIsolated(const TrackingRecHit *therechit, const edm::Event &evt) const;

  /// filter the n highest pt tracks
  Tracks theNHighestPtTracks(const Tracks &tracks) const;

  /// compare two tracks in pt (used by theNHighestPtTracks)
  struct ComparePt {
    bool operator()(const reco::Track *t1, const reco::Track *t2) const { return t1->pt() > t2->pt(); }
  };
  ComparePt ptComparator;

  const bool applyBasicCuts_, applyNHighestPt_, applyMultiplicityFilter_;
  const int seedOnlyFromAbove_;
  const bool applyIsolation_, chargeCheck_;
  const int nHighestPt_, minMultiplicity_, maxMultiplicity_;
  const bool multiplicityOnInput_;  /// if true, cut min/maxMultiplicity on input
                                    /// instead of on final result
  const double ptMin_, ptMax_, etaMin_, etaMax_, phiMin_, phiMax_, nHitMin_, nHitMax_, chi2nMax_;
  const double minHitChargeStrip_, minHitIsolation_;
  const edm::InputTag rphirecHitsTag_;
  const edm::InputTag matchedrecHitsTag_;
  const unsigned int nHitMin2D_;
  const int minHitsinTIB_, minHitsinTOB_, minHitsinTID_, minHitsinTEC_, minHitsinBPIX_, minHitsinFPIX_;
  const edm::EDGetTokenT<SiStripRecHit2DCollection> rphirecHitsToken_;
  const edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> matchedrecHitsToken_;
};

#endif
