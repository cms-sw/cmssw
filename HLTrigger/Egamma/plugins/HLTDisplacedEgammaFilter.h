#ifndef HLTrigger_Egamma_HLTDisplacedEgammaFilter_h
#define HLTrigger_Egamma_HLTDisplacedEgammaFilter_h

/** \class HLTDisplacedEgammaFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTDisplacedEgammaFilter : public HLTFilter {
public:
  explicit HLTDisplacedEgammaFilter(const edm::ParameterSet&);
  ~HLTDisplacedEgammaFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  edm::InputTag inputTag_;  // input tag identifying product contains egammas
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> inputToken_;
  int ncandcut_;  // number of egammas required
  edm::InputTag l1EGTag_;
  edm::InputTag rechitsEB;
  edm::InputTag rechitsEE;
  edm::EDGetTokenT<EcalRecHitCollection> rechitsEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> rechitsEEToken_;

  bool EBOnly;
  double sMin_min;
  double sMin_max;
  double sMaj_min;
  double sMaj_max;
  double seedTimeMin;
  double seedTimeMax;
  bool useTrackVeto;

  edm::InputTag inputTrk;
  edm::EDGetTokenT<reco::TrackCollection> inputTrkToken_;
  double trkPtCut;
  double trkDr2Cut;
  int maxTrkCut;
};

#endif
