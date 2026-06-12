// Author: Felice Pantaleo - CERN
// Flat-table dump of reco::PFRecHit collections (barrel/forward calorimeters:
// ECAL, HBHE, HF, HO). HGCal rechits are dumped separately by
// RecHitFlatTableProducer.
//
// NOTE: reco::PFRecHit::position()/positionREP() read the cached CaloCellGeometry,
// which is NOT persisted; calling them on rechits read back from a file segfaults.
// Positions are therefore recomputed from CaloGeometry using the (persisted) detId.

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
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <string>
#include <vector>

class PFRecHitFlatTableProducer : public edm::stream::EDProducer<> {
public:
  explicit PFRecHitFlatTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        rechits_tokens_{edm::vector_transform(
            params.getParameter<std::vector<edm::InputTag>>("label_rechits"),
            [this](edm::InputTag const& tag) { return consumes<reco::PFRecHitCollection>(tag); })},
        geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {
    produces<nanoaod::FlatTable>();
  }

  void produce(edm::Event& event, edm::EventSetup const& setup) override {
    auto const& geom = setup.getData(geomToken_);

    std::vector<uint32_t> rechit_ID;
    std::vector<float> rechit_energy;
    std::vector<float> rechit_time;
    std::vector<float> rechit_x;
    std::vector<float> rechit_y;
    std::vector<float> rechit_z;
    std::vector<float> rechit_eta;
    std::vector<float> rechit_phi;
    std::vector<int> rechit_depth;

    for (auto const& token : rechits_tokens_) {
      edm::Handle<reco::PFRecHitCollection> handle;
      event.getByToken(token, handle);
      if (!handle.isValid())
        continue;

      for (auto const& rh : *handle) {
        const GlobalPoint pos = geom.getPosition(DetId(rh.detId()));
        rechit_ID.push_back(rh.detId());
        rechit_energy.push_back(rh.energy());
        rechit_time.push_back(rh.time());
        rechit_x.push_back(pos.x());
        rechit_y.push_back(pos.y());
        rechit_z.push_back(pos.z());
        rechit_eta.push_back(pos.eta());
        rechit_phi.push_back(pos.phi());
        rechit_depth.push_back(rh.depth());
      }
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(rechit_ID.size(), objName_, false, false);
    tab->addColumn<uint32_t>("rechit_ID", rechit_ID, "PFRecHit DetId rawId");
    tab->addColumn<float>("rechit_energy", rechit_energy, "PFRecHit energy [GeV]");
    tab->addColumn<float>("rechit_time", rechit_time, "PFRecHit time [ns]");
    tab->addColumn<float>("rechit_x", rechit_x, "Global x from CaloGeometry [cm]");
    tab->addColumn<float>("rechit_y", rechit_y, "Global y from CaloGeometry [cm]");
    tab->addColumn<float>("rechit_z", rechit_z, "Global z from CaloGeometry [cm]");
    tab->addColumn<float>("rechit_eta", rechit_eta, "PFRecHit eta");
    tab->addColumn<float>("rechit_phi", rechit_phi, "PFRecHit phi");
    tab->addColumn<int>("rechit_depth", rechit_depth, "PFRecHit depth");

    event.put(std::move(tab));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("objName", "pfrechits")
        ->setComment("name of the nanoaod::FlatTable produced for barrel/forward calo PFRecHits");
    desc.add<std::vector<edm::InputTag>>("label_rechits",
                                         {edm::InputTag("hltParticleFlowRecHitECALUnseeded", "", "HLT"),
                                          edm::InputTag("hltParticleFlowRecHitHBHE", "", "HLT"),
                                          edm::InputTag("hltParticleFlowRecHitHF", "", "HLT"),
                                          edm::InputTag("hltParticleFlowRecHitHO", "", "HLT")})
        ->setComment("reco::PFRecHit collections to dump (barrel/forward calorimeters)");
    descriptions.add("pfRecHitTable", desc);
  }

private:
  const std::string objName_;
  const std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> rechits_tokens_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
};

DEFINE_FWK_MODULE(PFRecHitFlatTableProducer);
