#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/moduleAbilities.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include <vector>
#include <iostream>

class RecHitFlatTableProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  RecHitFlatTableProducer(edm::ParameterSet const& params)
      : objName_(params.getParameter<std::string>("objName")),
        rechits_tokens_{
            edm::vector_transform(params.getParameter<std::vector<edm::InputTag>>("label_rechits"),
                                  [this](const edm::InputTag& lab) { return consumes<HGCRecHitCollection>(lab); })},
        caloGeometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()) {
    produces<nanoaod::FlatTable>();
  }

  ~RecHitFlatTableProducer() override {}

  void produce(edm::Event& event, edm::EventSetup const& iSetup) override {
    std::vector<uint32_t> rechit_ID;
    std::vector<float> rechit_energy;
    std::vector<float> rechit_x;
    std::vector<float> rechit_y;
    std::vector<float> rechit_z;
    std::vector<float> rechit_time;
    std::vector<float> rechit_radius;
    std::vector<float> rechit_simEnergy;
    std::vector<float> rechit_simEnergyEM;
    std::vector<float> rechit_simEnergyHad;

    for (auto const& rh_token : rechits_tokens_) {
      edm::Handle<HGCRecHitCollection> rechit_handle;
      event.getByToken(rh_token, rechit_handle);
      const auto& rhColl = *rechit_handle;
      for (auto const& rh : rhColl) {
        rechit_energy.push_back(rh.energy());
        auto const rhPosition = rhtools_.getPosition(rh.detid());
        rechit_x.push_back(rhPosition.x());
        rechit_y.push_back(rhPosition.y());
        rechit_z.push_back(rhPosition.z());
        rechit_ID.push_back(rh.detid().rawId());
        rechit_time.push_back(rh.time());
        rechit_radius.push_back(rhtools_.getRadiusToSide(rh.detid()));
        // const auto hitId = hitMap->find(DetId(rh.detid()));
        // if (hitId != hitMap->end()) {
        //   rechit_simEnergy.push_back(hitIdToEnergies[hitId->second].energy);
        //   rechit_simEnergyEM.push_back(hitIdToEnergies[hitId->second].energyEM);
        //   rechit_simEnergyHad.push_back(hitIdToEnergies[hitId->second].energyHad);
        // }
      }
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(rechit_ID.size(), objName_, false, false);
    tab->addColumn<uint32_t>("rechit_ID", rechit_ID, "Rechit ID");
    tab->addColumn<float>("rechit_x", rechit_x, "Rechit X from rechittools");
    tab->addColumn<float>("rechit_y", rechit_y, "Rechit Y from rechittools");
    tab->addColumn<float>("rechit_z", rechit_z, "Rechit Z from rechittools");
    tab->addColumn<float>("rechit_radius", rechit_radius, "Rechit radius to side from rechittools");

    event.put(std::move(tab));
  }

  void beginRun(edm::Run const&, edm::EventSetup const& es) override {
    edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeometry_token_);
    rhtools_.setGeometry(*geom);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("objName", "rechits")->setComment("name of the nanoaod::FlatTable to extend with this table");
    desc.add<std::vector<edm::InputTag>>("label_rechits",
                                         {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                          edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
    descriptions.add("recHitTable", desc);
  }

protected:
  const std::string objName_;
  //   const edm::EDGetTokenT<edm::View<pat::PackedGenParticle>> src_;
  const std::vector<edm::EDGetTokenT<HGCRecHitCollection>> rechits_tokens_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_token_;
  hgcal::RecHitTools rhtools_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecHitFlatTableProducer);
