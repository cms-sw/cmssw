// A table of the rechits of one ECAL (EB, EE, ES) or HCAL (HBHE, HF, HO) collection, one row per hit in collection
// order: raw DetId, energy, time, and the global position of the cell from CaloGeometry (as used for PF rechits).
// Other tables can join on the DetId, or add columns as an extension table with the same src and cut.

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

template <typename C>
class CaloRecHitTableProducer final : public edm::global::EDProducer<> {
  using Hit = typename C::value_type;

public:
  explicit CaloRecHitTableProducer(const edm::ParameterSet& cfg)
      : name_(cfg.getParameter<std::string>("name")),
        doc_(cfg.getParameter<std::string>("doc")),
        precision_(cfg.getParameter<int>("precision")),
        skipNonExistingSrc_(cfg.getParameter<bool>("skipNonExistingSrc")),
        cut_(cfg.getParameter<std::string>("cut")),
        hitsToken_(consumes(cfg.getParameter<edm::InputTag>("src"))),
        geometryToken_(esConsumes()),
        tableToken_(produces()) {}

  void produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const override {
    std::vector<uint32_t> detIds;
    std::vector<float> energies, times, xs, ys, zs;
    const auto hits = event.getHandle(hitsToken_);
    if (hits.isValid() || !skipNonExistingSrc_) {  // dereferencing an invalid handle throws
      const auto& geometry = setup.getData(geometryToken_);
      // capacity to reserve: exact without a cut, an upper bound with one
      const std::size_t n = hits->size();
      detIds.reserve(n);
      for (auto* column : {&energies, &times, &xs, &ys, &zs})
        column->reserve(n);
      for (const Hit& hit : *hits) {
        if (!cut_(hit))
          continue;
        const DetId id = hit.detid();
        const auto cell = geometry.getGeometry(id);
        if (cell == nullptr)
          throw cms::Exception("CaloRecHitTableProducer")
              << "no cell in the calorimeter geometry for the rechit of DetId " << id.rawId()
              << ": is the geometry that of the input?";
        const auto& position = cell->getPosition();
        detIds.push_back(id.rawId());
        energies.push_back(hit.energy());
        times.push_back(hit.time());
        xs.push_back(position.x());
        ys.push_back(position.y());
        zs.push_back(position.z());
      }
    }

    auto table = std::make_unique<nanoaod::FlatTable>(detIds.size(), name_, false);
    table->setDoc(doc_);
    table->addColumn<uint32_t>("detId", detIds, "raw DetId of the hit");
    table->addColumn<float>("energy", energies, "energy [GeV]", precision_);
    table->addColumn<float>("time", times, "time [ns]", precision_);
    table->addColumn<float>("x", xs, "global x of the cell's reference point [cm]", precision_);
    table->addColumn<float>("y", ys, "global y of the cell's reference point [cm]", precision_);
    table->addColumn<float>("z", zs, "global z of the cell's reference point [cm]", precision_);
    event.put(tableToken_, std::move(table));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src")->setComment("input rechit collection");
    desc.add<std::string>("cut", "")->setComment("selection of the hits (string expression on the rechit class)");
    desc.add<std::string>("name")->setComment("name of the table");
    desc.add<std::string>("doc", "")->setComment("documentation of the table");
    desc.add<int>("precision", -1)->setComment("mantissa bits kept in the float columns (zero or negative: all)");
    desc.add<bool>("skipNonExistingSrc", false)->setComment("empty table for a missing collection instead of failing");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const std::string name_, doc_;
  const int precision_;
  const bool skipNonExistingSrc_;
  const StringCutObjectSelector<Hit> cut_;
  const edm::EDGetTokenT<C> hitsToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  const edm::EDPutTokenT<nanoaod::FlatTable> tableToken_;
};

using EcalRecHitTableProducer = CaloRecHitTableProducer<EcalRecHitCollection>;
using HBHERecHitTableProducer = CaloRecHitTableProducer<HBHERecHitCollection>;
using HFRecHitTableProducer = CaloRecHitTableProducer<HFRecHitCollection>;
using HORecHitTableProducer = CaloRecHitTableProducer<HORecHitCollection>;
DEFINE_FWK_MODULE(EcalRecHitTableProducer);
DEFINE_FWK_MODULE(HBHERecHitTableProducer);
DEFINE_FWK_MODULE(HFRecHitTableProducer);
DEFINE_FWK_MODULE(HORecHitTableProducer);
