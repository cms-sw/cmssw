//
// Class:           HITRegionalPixelSeedGenerator

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"

class HITRegionalPixelSeedGenerator : public TrackingRegionProducer {
public:
  explicit HITRegionalPixelSeedGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC)
      : m_regionPSet(conf.getParameter<edm::ParameterSet>("RegionPSet")),
        ptmin(m_regionPSet.getParameter<double>("ptMin")),
        originradius(m_regionPSet.getParameter<double>("originRadius")),
        halflength(m_regionPSet.getParameter<double>("originHalfLength")),
        etaCenter_(m_regionPSet.getParameter<double>("etaCenter")),
        phiCenter_(m_regionPSet.getParameter<double>("phiCenter")),
        deltaTrackEta(m_regionPSet.getParameter<double>("deltaEtaTrackRegion")),
        deltaTrackPhi(m_regionPSet.getParameter<double>("deltaPhiTrackRegion")),
        deltaL1JetEta(m_regionPSet.getParameter<double>("deltaEtaL1JetRegion")),
        deltaL1JetPhi(m_regionPSet.getParameter<double>("deltaPhiL1JetRegion")),
        usejets_(m_regionPSet.getParameter<bool>("useL1Jets")),
        usetracks_(m_regionPSet.getParameter<bool>("useTracks")),
        fixedReg_(m_regionPSet.getParameter<bool>("fixedReg")),
        useIsoTracks_(m_regionPSet.getParameter<bool>("useIsoTracks")),
        token_bfield(iC.esConsumes()),
        token_msmaker(iC.esConsumes()) {
    edm::LogVerbatim("HITRegionalPixelSeedGenerator") << "Enter the HITRegionalPixelSeedGenerator";

    if (usetracks_)
      token_trks = iC.consumes<reco::TrackCollection>(m_regionPSet.getParameter<edm::InputTag>("trackSrc"));
    if (usetracks_ || useIsoTracks_ || fixedReg_ || usejets_)
      token_vertex = iC.consumes<reco::VertexCollection>(m_regionPSet.getParameter<edm::InputTag>("vertexSrc"));
    if (useIsoTracks_)
      token_isoTrack =
          iC.consumes<trigger::TriggerFilterObjectWithRefs>(m_regionPSet.getParameter<edm::InputTag>("isoTrackSrc"));
    if (usejets_)
      token_l1jet =
          iC.consumes<l1extra::L1JetParticleCollection>(m_regionPSet.getParameter<edm::InputTag>("l1tjetSrc"));
  }

  ~HITRegionalPixelSeedGenerator() override = default;

  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& e, const edm::EventSetup& es) const override {
    std::vector<std::unique_ptr<TrackingRegion> > result;
    float originz = 0.;

    double deltaZVertex = halflength;

    auto const& bfield = es.getData(token_bfield);
    auto const& msmaker = es.getData(token_msmaker);

    if (usetracks_) {
      const edm::Handle<reco::TrackCollection>& tracks = e.getHandle(token_trks);

      const reco::VertexCollection& vertCollection = e.get(token_vertex);
      reco::VertexCollection::const_iterator ci = vertCollection.begin();

      if (!vertCollection.empty()) {
        originz = ci->z();
      } else {
        deltaZVertex = 15.;
      }

      GlobalVector globalVector(0, 0, 1);
      if (tracks->empty())
        return result;

      reco::TrackCollection::const_iterator itr = tracks->begin();
      for (; itr != tracks->end(); itr++) {
        GlobalVector ptrVec((itr)->px(), (itr)->py(), (itr)->pz());
        globalVector = ptrVec;

        result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(globalVector,
                                                                           GlobalPoint(0, 0, originz),
                                                                           ptmin,
                                                                           originradius,
                                                                           deltaZVertex,
                                                                           deltaTrackEta,
                                                                           deltaTrackPhi,
                                                                           bfield,
                                                                           &msmaker));
      }
    }

    if (useIsoTracks_) {
      const edm::Handle<trigger::TriggerFilterObjectWithRefs>& isotracks = e.getHandle(token_isoTrack);

      std::vector<edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;

      isotracks->getObjects(trigger::TriggerTrack, isoPixTrackRefs);

      const reco::VertexCollection& vertCollection = e.get(token_vertex);
      reco::VertexCollection::const_iterator ci = vertCollection.begin();

      if (!vertCollection.empty()) {
        originz = ci->z();
      } else {
        deltaZVertex = 15.;
      }

      GlobalVector globalVector(0, 0, 1);
      if (isoPixTrackRefs.empty())
        return result;

      for (uint32_t p = 0; p < isoPixTrackRefs.size(); p++) {
        GlobalVector ptrVec((isoPixTrackRefs[p]->track())->px(),
                            (isoPixTrackRefs[p]->track())->py(),
                            (isoPixTrackRefs[p]->track())->pz());
        globalVector = ptrVec;

        result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(globalVector,
                                                                           GlobalPoint(0, 0, originz),
                                                                           ptmin,
                                                                           originradius,
                                                                           deltaZVertex,
                                                                           deltaTrackEta,
                                                                           deltaTrackPhi,
                                                                           bfield,
                                                                           &msmaker));
      }
    }

    if (usejets_) {
      const edm::Handle<l1extra::L1JetParticleCollection>& jets = e.getHandle(token_l1jet);
      const reco::VertexCollection& vertCollection = e.get(token_vertex);
      reco::VertexCollection::const_iterator ci = vertCollection.begin();
      if (!vertCollection.empty()) {
        originz = ci->z();
      } else {
        deltaZVertex = 15.;
      }

      GlobalVector globalVector(0, 0, 1);
      if (jets->empty())
        return result;

      for (l1extra::L1JetParticleCollection::const_iterator iJet = jets->begin(); iJet != jets->end(); iJet++) {
        GlobalVector jetVector(iJet->p4().x(), iJet->p4().y(), iJet->p4().z());
        GlobalPoint vertex(0, 0, originz);

        result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(
            jetVector, vertex, ptmin, originradius, deltaZVertex, deltaL1JetEta, deltaL1JetPhi, bfield, &msmaker));
      }
    }
    if (fixedReg_) {
      GlobalVector fixedVector(cos(phiCenter_) * sin(2 * atan(exp(-etaCenter_))),
                               sin(phiCenter_) * sin(2 * atan(exp(-etaCenter_))),
                               cos(2 * atan(exp(-etaCenter_))));
      GlobalPoint vertex(0, 0, originz);

      const reco::VertexCollection& vertCollection = e.get(token_vertex);
      if (!vertCollection.empty()) {
        //      reco::VertexCollection::const_iterator ci = vertCollection.begin();
        //      originz = ci->z();
      } else {
        deltaZVertex = 15.;
      }

      result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(
          fixedVector, vertex, ptmin, originradius, deltaZVertex, deltaL1JetEta, deltaL1JetPhi, bfield, &msmaker));
    }

    return result;
  }

private:
  const edm::ParameterSet m_regionPSet;
  const float ptmin;
  const float originradius;
  const float halflength;
  const double etaCenter_;
  const double phiCenter_;
  const float deltaTrackEta;
  const float deltaTrackPhi;
  const float deltaL1JetEta;
  const float deltaL1JetPhi;
  const bool usejets_;
  const bool usetracks_;
  const bool fixedReg_;
  const bool useIsoTracks_;
  edm::EDGetTokenT<reco::TrackCollection> token_trks;
  edm::EDGetTokenT<reco::VertexCollection> token_vertex;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> token_isoTrack;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> token_l1jet;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> token_bfield;
  const edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> token_msmaker;
};

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#
DEFINE_EDM_PLUGIN(TrackingRegionProducerFactory, HITRegionalPixelSeedGenerator, "HITRegionalPixelSeedGenerator");
