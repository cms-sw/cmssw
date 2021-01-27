// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"


#include "DataFormats/Common/interface/Association.h"

#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include <set>

//
// class decleration
//
typedef std::pair<size_t, float> IdxAndFraction;

class SimClusterRecHitAssociationProducer : public edm::stream::EDProducer<> {
public:
  explicit SimClusterRecHitAssociationProducer(const edm::ParameterSet &);
  ~SimClusterRecHitAssociationProducer() override;

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<edm::InputTag> caloRechitTags_;
  std::vector<edm::EDGetTokenT<edm::PCaloHitContainer>> caloSimhitCollectionTokens_;
  //std::vector<edm::EDGetTokenT<std::vector<PSimHit>> trackSimhitCollectionTokens_;
  std::vector<edm::EDGetTokenT<edm::View<CaloRecHit>>> caloRechitCollectionTokens_;
  edm::EDGetTokenT<SimClusterCollection> scCollectionToken_;
};

SimClusterRecHitAssociationProducer::SimClusterRecHitAssociationProducer(const edm::ParameterSet &pset)
    : //caloSimhitCollectionTokens_(edm::vector_transform(pset.getParameter<std::vector<edm::InputTag>>("caloSimHits"),
      //  [this](const edm::InputTag& tag) {return mayConsume<edm::PCaloHitContainer>(tag); })),
    //trackSimhitCollectionTokens_(edm::vector_transform(pset.getParameter<edm::InputTag>("trackSimHits"),
    //    [this](const edm::InputTag& tag) {return mayConsume<std::vector<PSimHit>(tag); }),
    caloRechitTags_(pset.getParameter<std::vector<edm::InputTag>>("caloRecHits")),
    caloRechitCollectionTokens_(edm::vector_transform(caloRechitTags_,
        [this](const edm::InputTag& tag) {return mayConsume<edm::View<CaloRecHit>>(tag); })),
    scCollectionToken_(consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("simClusters")))
{
  for (auto& tag : caloRechitTags_) {
    std::string label = tag.instance();
    produces<edm::Association<SimClusterCollection>>(label+"ToSimClus");
  }
  produces<std::unordered_map<int, float>>();
}

SimClusterRecHitAssociationProducer::~SimClusterRecHitAssociationProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SimClusterRecHitAssociationProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  auto simClusterToRecEnergy = std::make_unique<std::unordered_map<int, float>>();
  edm::Handle<SimClusterCollection> scCollection;
  iEvent.getByToken(scCollectionToken_, scCollection);
  std::unordered_map<size_t, IdxAndFraction> hitDetIdToIndex;
  std::unordered_map<size_t, float> hitDetIdToTotalSimFrac;

  for (size_t s = 0; s < scCollection->size(); s++) {
    const auto& sc = scCollection->at(s);
    (*simClusterToRecEnergy)[s] = 0.;
    for (auto& hf : sc.hits_and_fractions()) {
        auto entry = hitDetIdToIndex.find(hf.first);
        // Update SimCluster assigment if detId has been found in no other SCs or if
        // SC has greater fraction of energy in DetId than the SC already found
        if (entry == hitDetIdToIndex.end()) {
            hitDetIdToTotalSimFrac[hf.first] = hf.second;
            hitDetIdToIndex[hf.first] = {s, hf.second};
        }
        else {
            hitDetIdToTotalSimFrac[hf.first] += hf.second;
            if (entry->second.second < hf.second)
                hitDetIdToIndex[hf.first] = {s, hf.second};
        }
    }
  }

  std::unordered_map<int, float> hitDetIdToTotalRecEnergy;

  for (size_t i = 0; i < caloRechitCollectionTokens_.size(); i++) {
    std::string label = caloRechitTags_.at(i).instance();
    std::vector<size_t> rechitIndices;

    edm::Handle<edm::View<CaloRecHit>> caloRechitCollection;
    iEvent.getByToken(caloRechitCollectionTokens_.at(i), caloRechitCollection);

    for (size_t h = 0; h < caloRechitCollection->size(); h++) {
        const CaloRecHit& caloRh = caloRechitCollection->at(h);
        size_t id = caloRh.detid().rawId();
        auto entry = hitDetIdToTotalRecEnergy.find(id);
        float energy = caloRh.energy();
        if (entry == hitDetIdToTotalRecEnergy.end())
            hitDetIdToTotalRecEnergy[id] = energy;
        else
            hitDetIdToTotalRecEnergy.at(id) += energy;

        int match = hitDetIdToIndex.find(id) == hitDetIdToIndex.end() ? -1 : hitDetIdToIndex.at(id).first;
        float fraction = match != -1 ? hitDetIdToTotalSimFrac.at(id) : 1.;
        if (simClusterToRecEnergy->find(match) == simClusterToRecEnergy->end())
            (*simClusterToRecEnergy)[match] = energy*fraction;
        else
            simClusterToRecEnergy->at(match) += energy*fraction;

        rechitIndices.push_back(match);
    }

    auto assoc = std::make_unique<edm::Association<SimClusterCollection>>(scCollection);
    edm::Association<SimClusterCollection>::Filler filler(*assoc);
    filler.insert(caloRechitCollection, rechitIndices.begin(), rechitIndices.end());
    filler.fill();
    iEvent.put(std::move(assoc), label+"ToSimClus");
  }
  iEvent.put(std::move(simClusterToRecEnergy));
}

// define this as a plug-in
DEFINE_FWK_MODULE(SimClusterRecHitAssociationProducer);

