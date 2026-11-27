// NanoAOD tables of the hits of calorimeter clusters, from hitsAndFractions(): a column n<hitsName> added to the
// cluster table, and a table <hitsName> with one row per (cluster, hit) entry. A hit can appear more than once in a
// SuperCluster, so its fraction in the cluster is the sum over its entries.
// Clusters are read as std::vector<T>, as an edm::View<reco::CaloCluster> is ambiguous when superclusters and their
// basic clusters share a label.

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

template <typename T>
class ClusterHitsTableProducer final : public edm::global::EDProducer<> {
public:
  explicit ClusterHitsTableProducer(const edm::ParameterSet& cfg)
      : name_(cfg.getParameter<std::string>("name")),
        hitsName_(cfg.getParameter<std::string>("hitsName")),
        hitsDoc_(cfg.getParameter<std::string>("hitsDoc")),
        precision_(cfg.getParameter<int>("precision")),
        maxLen_(cfg.existsAs<unsigned int>("maxLen") ? cfg.getParameter<unsigned int>("maxLen")
                                                     : std::numeric_limits<unsigned int>::max()),
        skipNonExistingSrc_(cfg.getParameter<bool>("skipNonExistingSrc")),
        cut_(cfg.getParameter<std::string>("cut")),
        clustersToken_(consumes(cfg.getParameter<edm::InputTag>("src"))),
        countsToken_(produces()),
        hitsToken_(produces(hitsName_ + "Table")) {
    if (hitsName_ == name_)
      throw cms::Exception("Configuration") << "hitsName must differ from the name of the cluster table, " << name_;
  }

  void produce(edm::StreamID, edm::Event& event, const edm::EventSetup&) const override {
    // countsTable extends the same name_ table of SimpleFlatTableProducer, so this must
    // select the same rows (cut, maxLen and skipNonExistingSrc handling)
    std::vector<const T*> selected;
    std::size_t nEntries = 0;
    auto const clusters = event.getHandle(clustersToken_);
    if (clusters.isValid() || !skipNonExistingSrc_) {
      selected.reserve(clusters->size());
      for (const auto& cluster : *clusters) {
        if (selected.size() >= maxLen_)
          break;
        if (cut_(cluster)) {
          selected.push_back(&cluster);
          nEntries += cluster.hitsAndFractions().size();
        }
      }
    }

    std::vector<uint16_t> counts;
    std::vector<uint32_t> detIds;
    std::vector<float> fractions;
    counts.reserve(selected.size());
    detIds.reserve(nEntries);
    fractions.reserve(nEntries);
    for (const T* cluster : selected) {
      const auto& hits = cluster->hitsAndFractions();
      // uint16, as the counts of the collection tables of SimpleCollectionFlatTableProducer
      if (hits.size() > std::numeric_limits<uint16_t>::max())
        throw cms::Exception("ClusterHitsTableProducer")
            << "a cluster of " << hits.size() << " hits is too large for the n" << hitsName_ << " column";
      counts.push_back(static_cast<uint16_t>(hits.size()));
      for (const auto& [id, fraction] : hits) {
        detIds.push_back(id.rawId());
        fractions.push_back(fraction);
      }
    }

    auto countsTable = std::make_unique<nanoaod::FlatTable>(counts.size(), name_, false, true);
    countsTable->addColumn<uint16_t>(
        "n" + hitsName_, counts, "number of hit entries of the cluster (rows in " + hitsName_ + ")");
    auto hitsTable = std::make_unique<nanoaod::FlatTable>(detIds.size(), hitsName_, false);
    hitsTable->setDoc(hitsDoc_);
    hitsTable->addColumn<uint32_t>("detId", detIds, "raw DetId of the hit");
    hitsTable->addColumn<float>(
        "fraction",
        fractions,
        "fraction of the hit energy in this entry (sum over the entries of a hit for its fraction in the cluster)",
        precision_);
    event.put(countsToken_, std::move(countsTable));
    event.put(hitsToken_, std::move(hitsTable));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src")->setComment("input cluster collection, a std::vector of the cluster type");
    desc.add<std::string>("cut", "")->setComment("selection of the clusters: must be that of the cluster table");
    desc.addOptional<unsigned int>("maxLen")->setComment(
        "maximum number of selected clusters: must be that of the cluster table");
    desc.add<std::string>("name")->setComment("name of the cluster table, extended with the n<hitsName> column");
    desc.add<std::string>("hitsName")
        ->setComment(
            "name of the table of the hits, one row per (cluster, hit) entry; n<hitsName> must not be a column of the "
            "cluster table already");
    desc.add<std::string>("hitsDoc", "")->setComment("documentation of the table of the hits");
    desc.add<int>("precision", -1)->setComment("mantissa bits kept in the fraction (zero or negative: all)");
    desc.add<bool>("skipNonExistingSrc", false)->setComment("empty tables for missing clusters instead of failing");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const std::string name_, hitsName_, hitsDoc_;
  const int precision_;
  const unsigned int maxLen_;
  const bool skipNonExistingSrc_;
  const StringCutObjectSelector<T> cut_;
  const edm::EDGetTokenT<std::vector<T>> clustersToken_;
  const edm::EDPutTokenT<nanoaod::FlatTable> countsToken_, hitsToken_;
};

using CaloClusterHitsTableProducer = ClusterHitsTableProducer<reco::CaloCluster>;
using PFClusterHitsTableProducer = ClusterHitsTableProducer<reco::PFCluster>;
using SuperClusterHitsTableProducer = ClusterHitsTableProducer<reco::SuperCluster>;
DEFINE_FWK_MODULE(CaloClusterHitsTableProducer);
DEFINE_FWK_MODULE(PFClusterHitsTableProducer);
DEFINE_FWK_MODULE(SuperClusterHitsTableProducer);
