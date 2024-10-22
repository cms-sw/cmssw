// -*- C++ -*-
//
// Package:     RecoTauTag/HLTProducers
// Class  :     TrackingRegionsFromBeamSpotAndL2Tau
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 13 Sep 2022 21:13:17 GMT
//

// system include files

// user include files
#include "TrackingRegionsFromBeamSpotAndL2Tau.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"

DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory,
                  TrackingRegionsFromBeamSpotAndL2Tau,
                  "TrackingRegionsFromBeamSpotAndL2Tau");

TrackingRegionsFromBeamSpotAndL2Tau::TrackingRegionsFromBeamSpotAndL2Tau(const edm::ParameterSet& conf,
                                                                         edm::ConsumesCollector&& iC) {
  edm::LogInfo("TrackingRegionsFromBeamSpotAndL2Tau") << "Enter the TrackingRegionsFromBeamSpotAndL2Tau";

  edm::ParameterSet regionPSet = conf.getParameter<edm::ParameterSet>("RegionPSet");

  m_ptMin = regionPSet.getParameter<double>("ptMin");
  m_originRadius = regionPSet.getParameter<double>("originRadius");
  m_originHalfLength = regionPSet.getParameter<double>("originHalfLength");
  m_deltaEta = regionPSet.getParameter<double>("deltaEta");
  m_deltaPhi = regionPSet.getParameter<double>("deltaPhi");
  token_jet = iC.consumes<reco::CandidateView>(regionPSet.getParameter<edm::InputTag>("JetSrc"));
  m_jetMinPt = regionPSet.getParameter<double>("JetMinPt");
  m_jetMaxEta = regionPSet.getParameter<double>("JetMaxEta");
  m_jetMaxN = regionPSet.getParameter<int>("JetMaxN");
  token_beamSpot = iC.consumes<reco::BeamSpot>(regionPSet.getParameter<edm::InputTag>("beamSpot"));
  m_precise = regionPSet.getParameter<bool>("precise");

  if (regionPSet.exists("searchOpt"))
    m_searchOpt = regionPSet.getParameter<bool>("searchOpt");
  else
    m_searchOpt = false;

  m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::stringToUseMeasurementTracker(
      regionPSet.getParameter<std::string>("whereToUseMeasurementTracker"));
  if (m_whereToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
    token_measurementTracker =
        iC.consumes<MeasurementTrackerEvent>(regionPSet.getParameter<edm::InputTag>("measurementTrackerName"));
  }
  token_field = iC.esConsumes();
  if (m_precise) {
    token_msmaker = iC.esConsumes();
  }
}

void TrackingRegionsFromBeamSpotAndL2Tau::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<double>("ptMin", 5.0);
  desc.add<double>("originRadius", 0.2);
  desc.add<double>("originHalfLength", 24.0);
  desc.add<double>("deltaEta", 0.3);
  desc.add<double>("deltaPhi", 0.3);
  desc.add<edm::InputTag>("JetSrc", edm::InputTag("hltFilterL2EtCutDoublePFIsoTau25Trk5"));
  desc.add<double>("JetMinPt", 25.0);
  desc.add<double>("JetMaxEta", 2.1);
  desc.add<int>("JetMaxN", 10);
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<bool>("precise", true);
  desc.add<std::string>("howToUseMeasurementTracker", "Never");
  desc.add<edm::InputTag>("measurementTrackerName", edm::InputTag("MeasurementTrackerEvent"));

  // Only for backwards-compatibility
  edm::ParameterSetDescription descRegion;
  descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

  descriptions.add("trackingRegionsFromBeamSpotAndL2Tau", descRegion);
}

std::vector<std::unique_ptr<TrackingRegion> > TrackingRegionsFromBeamSpotAndL2Tau::regions(
    const edm::Event& e, const edm::EventSetup& es) const {
  std::vector<std::unique_ptr<TrackingRegion> > result;

  // use beam spot to pick up the origin
  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByToken(token_beamSpot, bsHandle);
  if (!bsHandle.isValid())
    return result;
  const reco::BeamSpot& bs = *bsHandle;
  GlobalPoint origin(bs.x0(), bs.y0(), bs.z0());

  // pick up the candidate objects of interest
  edm::Handle<reco::CandidateView> objects;
  e.getByToken(token_jet, objects);
  size_t n_objects = objects->size();
  if (n_objects == 0)
    return result;

  const MeasurementTrackerEvent* measurementTracker = nullptr;
  if (!token_measurementTracker.isUninitialized()) {
    edm::Handle<MeasurementTrackerEvent> hmte;
    e.getByToken(token_measurementTracker, hmte);
    measurementTracker = hmte.product();
  }

  const auto& field = es.getData(token_field);
  const MultipleScatteringParametrisationMaker* msmaker = nullptr;
  if (m_precise) {
    msmaker = &es.getData(token_msmaker);
  }

  // create maximum JetMaxN tracking regions in directions of
  // highest pt jets that are above threshold and are within allowed eta
  // (we expect that jet collection was sorted in decreasing pt order)
  int n_regions = 0;
  for (size_t i = 0; i < n_objects && n_regions < m_jetMaxN; ++i) {
    const reco::Candidate& jet = (*objects)[i];
    if (jet.pt() < m_jetMinPt || std::abs(jet.eta()) > m_jetMaxEta)
      continue;

    GlobalVector direction(jet.momentum().x(), jet.momentum().y(), jet.momentum().z());

    result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(direction,
                                                                       origin,
                                                                       m_ptMin,
                                                                       m_originRadius,
                                                                       m_originHalfLength,
                                                                       m_deltaEta,
                                                                       m_deltaPhi,
                                                                       field,
                                                                       msmaker,
                                                                       m_precise,
                                                                       m_whereToUseMeasurementTracker,
                                                                       measurementTracker,
                                                                       m_searchOpt));
    ++n_regions;
  }
  //std::cout<<"nregions = "<<n_regions<<std::endl;
  return result;
}
