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

class SimHitRecHitAssociationProducer : public edm::stream::EDProducer<> {
public:
  explicit SimHitRecHitAssociationProducer(const edm::ParameterSet &);
  ~SimHitRecHitAssociationProducer() override;

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<edm::InputTag> caloRechitTags_;
  std::vector<edm::EDGetTokenT<edm::PCaloHitContainer>> caloSimhitCollectionTokens_;
  //std::vector<edm::EDGetTokenT<std::vector<PSimHit>> trackSimhitCollectionTokens_;
  std::vector<edm::EDGetTokenT<edm::View<CaloRecHit>>> caloRechitCollectionTokens_;
  edm::EDGetTokenT<SimClusterCollection> scCollectionToken_;
};

SimHitRecHitAssociationProducer::SimHitRecHitAssociationProducer(const edm::ParameterSet &pset)
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
}

SimHitRecHitAssociationProducer::~SimHitRecHitAssociationProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SimHitRecHitAssociationProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<SimClusterCollection> scCollection;
  iEvent.getByToken(scCollectionToken_, scCollection);
  std::unordered_map<size_t, IdxAndFraction> hitDetIdToIndex;

  for (size_t s = 0; s < scCollection->size(); s++) {
    const auto& sc = scCollection->at(s);
    for (auto& hf : sc.hits_and_fractions()) {
        auto entry = hitDetIdToIndex.find(hf.first);
        // Update SimCluster assigment if detId has been found in no other SCs or if
        // SC has greater fraction of energy in DetId than the SC already found
        if (entry == hitDetIdToIndex.end() || entry->second.second < hf.second)
            hitDetIdToIndex[hf.first] = {s, hf.second};
    }
  }

  for (size_t i = 0; i < caloRechitCollectionTokens_.size(); i++) {
    std::string label = caloRechitTags_.at(i).instance();
    std::vector<size_t> rechitIndices;

    edm::Handle<edm::View<CaloRecHit>> caloRechitCollection;
    iEvent.getByToken(caloRechitCollectionTokens_.at(i), caloRechitCollection);
    //edm::Handle<edm::PCaloHitContainer> caloSimhitCollection;
    //iEvent.getByToken(caloSimhitCollectionTokens_.at(i), caloSimhitCollection);

    for (size_t h = 0; h < caloRechitCollection->size(); h++) {
        const CaloRecHit& caloRh = caloRechitCollection->at(h);
        size_t id = caloRh.detid().rawId();
        int match = hitDetIdToIndex.find(id) == hitDetIdToIndex.end() ? -1 : hitDetIdToIndex.at(id).first;
        rechitIndices.push_back(match);
    }

    auto assoc = std::make_unique<edm::Association<SimClusterCollection>>(scCollection);
    edm::Association<SimClusterCollection>::Filler filler(*assoc);
    filler.insert(caloRechitCollection, rechitIndices.begin(), rechitIndices.end());
    filler.fill();
    iEvent.put(std::move(assoc), label+"ToSimClus");
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(SimHitRecHitAssociationProducer);

