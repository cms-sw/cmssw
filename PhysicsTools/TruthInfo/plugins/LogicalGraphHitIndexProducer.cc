#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

namespace {

  struct LogicalGraphView {
    explicit LogicalGraphView(truth::Graph const& graph) : graph_(graph) {}

    uint32_t nParticles() const { return graph_.nParticles(); }

    bool particleHasSim(uint32_t particleId) const {
      return particleId < graph_.particles.size() && graph_.particles[particleId].hasSim();
    }

    int32_t particleSimNode(uint32_t particleId) const { return graph_.particles[particleId].simNode; }

    template <typename F>
    void forEachParticleChild(uint32_t parentParticleId, F&& f) const {
      if (parentParticleId >= graph_.nParticles())
        return;

      for (const uint32_t vertexId : graph_.decayVertices(parentParticleId)) {
        if (vertexId >= graph_.nVertices())
          continue;

        for (const uint32_t childId : graph_.outgoingParticles(vertexId)) {
          f(childId);
        }
      }
    }

    truth::Graph const& graph_;
  };

  uint32_t checkedTrackId(int64_t key) {
    if (key < 0 || key > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      throw std::runtime_error("Invalid SimTrack key in logical graph");

    return static_cast<uint32_t>(key);
  }

}  // namespace

class TruthLogicalGraphHitIndexProducer : public edm::global::EDProducer<> {
public:
  explicit TruthLogicalGraphHitIndexProducer(edm::ParameterSet const& cfg)
      : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
        rawGraphToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),

        simTracksToken_(consumes<edm::SimTrackContainer>(cfg.getParameter<edm::InputTag>("simTracks"))),
        simHitCollections_(cfg.getParameter<std::vector<edm::InputTag>>("simHitCollections")),
        doHGCal_(cfg.getParameter<bool>("doHGCal")),
        doHGCalRelabelling_(cfg.getParameter<bool>("doHGCalRelabelling")),
        geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {
    for (auto const& tag : simHitCollections_) {
      simHitTokens_.push_back(consumes<std::vector<PCaloHit>>(tag));
    }

    produces<truth::LogicalGraphHitIndex>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
    desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"));

    desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));

    desc.add<std::vector<edm::InputTag>>("simHitCollections",
                                         {
                                             edm::InputTag("g4SimHits", "HGCHitsEE"),
                                             edm::InputTag("g4SimHits", "HGCHitsHEfront"),
                                             edm::InputTag("g4SimHits", "HGCHitsHEback"),
                                         });

    desc.add<bool>("doHGCal", true);
    desc.add<bool>("doHGCalRelabelling", true);

    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event& event, edm::EventSetup const& setup) const override {
    auto const& graph = event.get(graphToken_);
    auto const& rawGraph = event.get(rawGraphToken_);

    LogicalGraphView graphView(graph);

    truth::LogicalGraphHitIndexBuilder builder(graphView.nParticles());

    fillTrackToParticleMap(graphView, rawGraph, builder);
    fillSimHits(event, setup, builder);

    auto output = std::make_unique<truth::LogicalGraphHitIndex>(builder.finish());
    event.put(std::move(output));
  }

private:
  void fillTrackToParticleMap(LogicalGraphView const& graph,
                              TruthGraph const& rawGraph,
                              truth::LogicalGraphHitIndexBuilder& builder) const {
    for (uint32_t particleId = 0; particleId < graph.nParticles(); ++particleId) {
      if (!graph.particleHasSim(particleId))
        continue;

      const int32_t simNode = graph.particleSimNode(particleId);
      if (simNode < 0)
        continue;

      const uint32_t simNodeU32 = static_cast<uint32_t>(simNode);
      if (simNodeU32 >= rawGraph.nNodes())
        continue;

      auto const& ref = rawGraph.nodeRef(simNodeU32);
      if (ref.kind != TruthGraph::NodeKind::SimTrack)
        continue;

      builder.setSimTrackForParticle(particleId, checkedTrackId(ref.key));
    }

    for (uint32_t parentId = 0; parentId < graph.nParticles(); ++parentId) {
      graph.forEachParticleChild(parentId, [&](uint32_t childId) { builder.addParticleChild(parentId, childId); });
    }
  }

  void fillSimHits(edm::Event const& event,
                   edm::EventSetup const& setup,
                   truth::LogicalGraphHitIndexBuilder& builder) const {
    GeometryCache geometry;
    if (doHGCalRelabelling_)
      geometry = makeGeometryCache(setup);

    for (std::size_t i = 0; i < simHitTokens_.size(); ++i) {
      edm::Handle<std::vector<PCaloHit>> hits;
      event.getByToken(simHitTokens_[i], hits);

      if (!hits.isValid())
        continue;

      const auto& tag = simHitCollections_[i];
      const bool isHcal = tag.instance().find("HcalHits") != std::string::npos;
      const bool isHGCal = tag.instance().find("HGCHits") != std::string::npos;

      for (auto const& hit : *hits) {
        if (hit.geantTrackId() == 0)
          continue;

        const DetId detId = makeRecoDetId(hit, isHGCal, isHcal, geometry);
        if (detId == DetId(0))
          continue;

        builder.addHitForTrack(hit.geantTrackId(), detId.rawId(), hit.energy());
      }
    }
  }

  struct GeometryCache {
    int geometryType = -1;

    HGCalTopology const* hgtopo[3] = {nullptr, nullptr, nullptr};
    HGCalDDDConstants const* hgddd[3] = {nullptr, nullptr, nullptr};

    HcalDDDRecConstants const* hcddd = nullptr;
  };

  GeometryCache makeGeometryCache(edm::EventSetup const& setup) const {
    GeometryCache cache;

    auto const& geom = setup.getData(geomToken_);

    auto const* hcalGeom = static_cast<HcalGeometry const*>(geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
    if (hcalGeom)
      cache.hcddd = hcalGeom->topology().dddConstants();

    if (!doHGCal_)
      return cache;

    auto const* eeGeom = static_cast<HGCalGeometry const*>(
        geom.getSubdetectorGeometry(DetId::HGCalEE, ForwardSubdetector::ForwardEmpty));

    if (eeGeom) {
      cache.geometryType = 1;

      auto const* fhGeom = static_cast<HGCalGeometry const*>(
          geom.getSubdetectorGeometry(DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty));
      auto const* bhGeom = static_cast<HGCalGeometry const*>(
          geom.getSubdetectorGeometry(DetId::HGCalHSc, ForwardSubdetector::ForwardEmpty));

      cache.hgtopo[0] = &eeGeom->topology();
      if (fhGeom)
        cache.hgtopo[1] = &fhGeom->topology();
      if (bhGeom)
        cache.hgtopo[2] = &bhGeom->topology();
    } else {
      cache.geometryType = 0;

      eeGeom = static_cast<HGCalGeometry const*>(geom.getSubdetectorGeometry(DetId::Forward, HGCEE));
      auto const* fhGeom = static_cast<HGCalGeometry const*>(geom.getSubdetectorGeometry(DetId::Forward, HGCHEF));

      if (eeGeom)
        cache.hgtopo[0] = &eeGeom->topology();
      if (fhGeom)
        cache.hgtopo[1] = &fhGeom->topology();
    }

    for (unsigned i = 0; i < 3; ++i) {
      if (cache.hgtopo[i])
        cache.hgddd[i] = &cache.hgtopo[i]->dddConstants();
    }

    return cache;
  }

  DetId makeRecoDetId(PCaloHit const& hit, bool isHGCal, bool isHcal, GeometryCache const& geometry) const {
    if (!doHGCalRelabelling_) {
      return DetId(hit.id());
    }

    if (isHGCal) {
      const uint32_t simId = hit.id();

      if (geometry.geometryType == 1) {
        return DetId(simId);
      }

      if (isHcal) {
        if (!geometry.hcddd)
          return DetId(0);

        HcalDetId hid = HcalHitRelabeller::relabel(simId, geometry.hcddd);
        if (hid.subdet() == HcalEndcap)
          return hid;

        return DetId(0);
      }

      int subdet = 0;
      int layer = 0;
      int cell = 0;
      int sec = 0;
      int subsec = 0;
      int zp = 0;

      HGCalTestNumbering::unpackHexagonIndex(simId, subdet, zp, layer, sec, subsec, cell);

      if (subdet < 3 || subdet > 5)
        return DetId(0);

      const unsigned idx = static_cast<unsigned>(subdet - 3);
      if (!geometry.hgddd[idx] || !geometry.hgtopo[idx])
        return DetId(0);

      auto const recoLayerCell = geometry.hgddd[idx]->simToReco(cell, layer, sec, geometry.hgtopo[idx]->detectorType());
      cell = recoLayerCell.first;
      layer = recoLayerCell.second;

      if (layer == -1)
        return DetId(0);

      return HGCalDetId(static_cast<ForwardSubdetector>(subdet), zp, layer, subsec, sec, cell);
    }

    if (isHcal) {
      if (!geometry.hcddd)
        return DetId(0);

      return HcalHitRelabeller::relabel(hit.id(), geometry.hcddd);
    }

    return DetId(hit.id());
  }

  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<TruthGraph> rawGraphToken_;

  edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;

  std::vector<edm::InputTag> simHitCollections_;
  std::vector<edm::EDGetTokenT<std::vector<PCaloHit>>> simHitTokens_;

  bool doHGCal_;
  bool doHGCalRelabelling_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

DEFINE_FWK_MODULE(TruthLogicalGraphHitIndexProducer);
