// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Author: Felice Pantaleo - CERN
// Flat-table dump of tracker PSimHit collections (g4SimHits TrackerHits*).
// Global positions are computed from the local PSimHit position using the
// TrackerGeometry. trackId() links each hit back to a SimTrack, i.e. to a
// truth-graph particle.

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include <string>
#include <vector>

class TrackerSimHitFlatTableProducer : public edm::stream::EDProducer<> {
public:
  explicit TrackerSimHitFlatTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        simhits_tokens_{
            edm::vector_transform(params.getParameter<std::vector<edm::InputTag>>("label_simhits"),
                                  [this](edm::InputTag const& tag) { return consumes<edm::PSimHitContainer>(tag); })},
        trackerGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()) {
    produces<nanoaod::FlatTable>();
  }

  void produce(edm::Event& event, edm::EventSetup const& setup) override {
    auto const& trackerGeom = setup.getData(trackerGeomToken_);

    std::vector<uint32_t> simhit_detId;
    std::vector<uint32_t> simhit_trackId;
    std::vector<int> simhit_pdgId;
    std::vector<float> simhit_energyLoss;
    std::vector<float> simhit_tof;
    std::vector<float> simhit_pabs;
    std::vector<int> simhit_processType;
    std::vector<float> simhit_x;
    std::vector<float> simhit_y;
    std::vector<float> simhit_z;

    for (auto const& token : simhits_tokens_) {
      edm::Handle<edm::PSimHitContainer> handle;
      event.getByToken(token, handle);
      if (!handle.isValid())
        continue;

      for (auto const& hit : *handle) {
        const DetId detId(hit.detUnitId());
        auto const* det = trackerGeom.idToDet(detId);

        GlobalPoint gp;
        if (det != nullptr)
          gp = det->surface().toGlobal(hit.localPosition());

        simhit_detId.push_back(hit.detUnitId());
        simhit_trackId.push_back(hit.trackId());
        simhit_pdgId.push_back(hit.particleType());
        simhit_energyLoss.push_back(hit.energyLoss());
        simhit_tof.push_back(hit.tof());
        simhit_pabs.push_back(hit.pabs());
        simhit_processType.push_back(hit.processType());
        simhit_x.push_back(gp.x());
        simhit_y.push_back(gp.y());
        simhit_z.push_back(gp.z());
      }
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(simhit_detId.size(), objName_, false, false);
    tab->addColumn<uint32_t>("simhit_detId", simhit_detId, "Tracker PSimHit detUnitId rawId");
    tab->addColumn<uint32_t>("simhit_trackId", simhit_trackId, "G4 trackId of the SimTrack that made the hit");
    tab->addColumn<int>("simhit_pdgId", simhit_pdgId, "PDG id of the particle (particleType)");
    tab->addColumn<float>("simhit_energyLoss", simhit_energyLoss, "Energy loss in the sensor [GeV]");
    tab->addColumn<float>("simhit_tof", simhit_tof, "Time of flight [ns]");
    tab->addColumn<float>("simhit_pabs", simhit_pabs, "Momentum magnitude at entry [GeV]");
    tab->addColumn<int>("simhit_processType", simhit_processType, "Geant process type");
    tab->addColumn<float>("simhit_x", simhit_x, "Global x of the hit center [cm]");
    tab->addColumn<float>("simhit_y", simhit_y, "Global y of the hit center [cm]");
    tab->addColumn<float>("simhit_z", simhit_z, "Global z of the hit center [cm]");

    event.put(std::move(tab));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("objName", "trackersimhits")
        ->setComment("name of the nanoaod::FlatTable produced for tracker PSimHits");
    desc.add<std::vector<edm::InputTag>>("label_simhits",
                                         {edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTIBLowTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTIBHighTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTIDLowTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTIDHighTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTOBLowTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTOBHighTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTECLowTof"),
                                          edm::InputTag("g4SimHits", "TrackerHitsTECHighTof")})
        ->setComment("Tracker PSimHit collections to dump");
    descriptions.add("trackerSimHitTable", desc);
  }

private:
  const std::string objName_;
  const std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simhits_tokens_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
};

DEFINE_FWK_MODULE(TrackerSimHitFlatTableProducer);
