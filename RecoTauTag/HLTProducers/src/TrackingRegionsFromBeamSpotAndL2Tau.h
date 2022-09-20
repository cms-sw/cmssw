#ifndef TrackingRegionsFromBeamSpotAndL2Tau_h
#define TrackingRegionsFromBeamSpotAndL2Tau_h

//
// Class:           TrackingRegionsFromBeamSpotAndL2Tau
//

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/Candidate.h"

/** class TrackingRegionsFromBeamSpotAndL2Tau
 * plugin for creating eta-phi TrackingRegions in directions of L2 taus
 */
class TrackingRegionsFromBeamSpotAndL2Tau : public TrackingRegionProducer {
public:
  explicit TrackingRegionsFromBeamSpotAndL2Tau(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);
  ~TrackingRegionsFromBeamSpotAndL2Tau() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& e, const edm::EventSetup& es) const override;

private:
  float m_ptMin;
  float m_originRadius;
  float m_originHalfLength;
  float m_deltaEta;
  float m_deltaPhi;
  edm::EDGetTokenT<reco::CandidateView> token_jet;
  float m_jetMinPt;
  float m_jetMaxEta;
  int m_jetMaxN;
  edm::EDGetTokenT<MeasurementTrackerEvent> token_measurementTracker;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_whereToUseMeasurementTracker;
  bool m_searchOpt;
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> token_field;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> token_msmaker;
  bool m_precise;
};

#endif
