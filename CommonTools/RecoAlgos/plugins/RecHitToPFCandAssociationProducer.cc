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
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/Common/interface/Association.h"

#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include <set>

//
// class decleration
//
typedef std::pair<size_t, float> IdxAndFraction;

class RecHitToPFCandAssociationProducer : public edm::stream::EDProducer<> {
public:
  explicit RecHitToPFCandAssociationProducer(const edm::ParameterSet &);
  ~RecHitToPFCandAssociationProducer() override;

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<edm::InputTag> caloRechitTags_;
  std::vector<edm::EDGetTokenT<edm::PCaloHitContainer>> caloSimhitCollectionTokens_;
  //std::vector<edm::EDGetTokenT<std::vector<PSimHit>> trackSimhitCollectionTokens_;
  std::vector<edm::EDGetTokenT<edm::View<CaloRecHit>>> caloRechitCollectionTokens_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCollectionToken_;
};

RecHitToPFCandAssociationProducer::RecHitToPFCandAssociationProducer(const edm::ParameterSet &pset)
    : //caloSimhitCollectionTokens_(edm::vector_transform(pset.getParameter<std::vector<edm::InputTag>>("caloSimHits"),
      //  [this](const edm::InputTag& tag) {return mayConsume<edm::PCaloHitContainer>(tag); })),
    //trackSimhitCollectionTokens_(edm::vector_transform(pset.getParameter<edm::InputTag>("trackSimHits"),
    //    [this](const edm::InputTag& tag) {return mayConsume<std::vector<PSimHit>(tag); }),
    caloRechitTags_(pset.getParameter<std::vector<edm::InputTag>>("caloRecHits")),
    caloRechitCollectionTokens_(edm::vector_transform(caloRechitTags_,
        [this](const edm::InputTag& tag) {return mayConsume<edm::View<CaloRecHit>>(tag); })),
    pfCollectionToken_(consumes<reco::PFCandidateCollection>(pset.getParameter<edm::InputTag>("pfCands")))
{
  for (auto& tag : caloRechitTags_) {
    std::string label = tag.instance();
    //TODO: Can this be an edm::View?
    produces<edm::Association<reco::PFCandidateCollection>>(label+"ToPFCand");
  }
}

RecHitToPFCandAssociationProducer::~RecHitToPFCandAssociationProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void RecHitToPFCandAssociationProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<reco::PFCandidateCollection> pfCollection;
  iEvent.getByToken(pfCollectionToken_, pfCollection);
  std::unordered_map<size_t, IdxAndFraction> hitDetIdToIndex;

  for (size_t j = 0; j < pfCollection->size(); j++) {
      const auto& pfCand = pfCollection->at(j);
      const reco::PFCandidate::ElementsInBlocks& elements = pfCand.elementsInBlocks();
      for (auto& element : elements) {
          const reco::PFBlockRef blockRef = element.first;
          for (const auto& block : blockRef->elements()) {
              if (block.type() == reco::PFBlockElement::HGCAL) {
                  const reco::PFClusterRef cluster = block.clusterRef();
                  const std::vector<reco::PFRecHitFraction>& rhf = cluster->recHitFractions();
                  for (const auto& hf : rhf) {
                      auto& hit = hf.recHitRef();
                      if (!hit)
                          throw cms::Exception("RecHitToPFCandAssociationProducer") << "Invalid RecHit ref";
                      size_t detId = hit->detId();
                      auto entry = hitDetIdToIndex.find(detId);
                      if (entry == hitDetIdToIndex.end() || entry->second.second < hf.fraction())
                        hitDetIdToIndex[detId] = {j, hf.fraction()};
                  }
              }
          }
      }
  }

  for (size_t i = 0; i < caloRechitCollectionTokens_.size(); i++) {
    std::string label = caloRechitTags_.at(i).instance();
    std::vector<size_t> rechitIndices;

    edm::Handle<edm::View<CaloRecHit>> caloRechitCollection;
    iEvent.getByToken(caloRechitCollectionTokens_.at(i), caloRechitCollection);

    for (size_t h = 0; h < caloRechitCollection->size(); h++) {
        const CaloRecHit& caloRh = caloRechitCollection->at(h);
        size_t id = caloRh.detid().rawId();
        int match = hitDetIdToIndex.find(id) == hitDetIdToIndex.end() ? -1 : hitDetIdToIndex.at(id).first;
        rechitIndices.push_back(match);
    }

    auto assoc = std::make_unique<edm::Association<reco::PFCandidateCollection>>(pfCollection);
    edm::Association<reco::PFCandidateCollection>::Filler filler(*assoc);
    filler.insert(caloRechitCollection, rechitIndices.begin(), rechitIndices.end());
    filler.fill();
    iEvent.put(std::move(assoc), label+"ToPFCand");
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(RecHitToPFCandAssociationProducer);


