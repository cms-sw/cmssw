#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitNavigatorBase.h"

#include <memory>

//
// class declaration
//

class PFRecHitProducer final : public edm::stream::EDProducer<> {
public:
  explicit PFRecHitProducer(const edm::ParameterSet& iConfig);
  ~PFRecHitProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, const edm::EventSetup&) override;
  std::vector<std::unique_ptr<PFRecHitCreatorBase> > creators_;
  std::unique_ptr<PFRecHitNavigatorBase> navigator_;
  bool init_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecHitProducer);

namespace {
  bool sortByDetId(const reco::PFRecHit& a, const reco::PFRecHit& b) { return a.detId() < b.detId(); }

  edm::RunningAverage localRA1;
  edm::RunningAverage localRA2;
}  // namespace

PFRecHitProducer::PFRecHitProducer(const edm::ParameterSet& iConfig) {
  produces<reco::PFRecHitCollection>();
  produces<reco::PFRecHitCollection>("Cleaned");

  edm::ConsumesCollector cc = consumesCollector();

  std::vector<edm::ParameterSet> creators = iConfig.getParameter<std::vector<edm::ParameterSet> >("producers");
  for (auto& creator : creators) {
    std::string name = creator.getParameter<std::string>("name");
    creators_.emplace_back(PFRecHitFactory::get()->create(name, creator, cc));
  }

  edm::ParameterSet navSet = iConfig.getParameter<edm::ParameterSet>("navigator");
  navigator_ = PFRecHitNavigationFactory::get()->create(navSet.getParameter<std::string>("name"), navSet, cc);
}

PFRecHitProducer::~PFRecHitProducer() = default;

//
// member functions
//

void PFRecHitProducer::beginRun(edm::Run const& iRun, const edm::EventSetup& iSetup) {
  for (const auto& creator : creators_) {
    creator->init(iSetup);
  }
  navigator_->init(iSetup);
}

// ------------ method called to produce the data  ------------
void PFRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  auto out = std::make_unique<reco::PFRecHitCollection>();
  auto cleaned = std::make_unique<reco::PFRecHitCollection>();

  out->reserve(localRA1.upper());
  cleaned->reserve(localRA2.upper());

  for (const auto& creator : creators_) {
    creator->importRecHits(out, cleaned, iEvent, iSetup);
  }

  if (out->capacity() > 2 * out->size())
    out->shrink_to_fit();
  if (cleaned->capacity() > 2 * cleaned->size())
    cleaned->shrink_to_fit();
  localRA1.update(out->size());
  localRA2.update(cleaned->size());
  std::sort(out->begin(), out->end(), sortByDetId);

  //create a refprod here
  edm::RefProd<reco::PFRecHitCollection> refProd = iEvent.getRefBeforePut<reco::PFRecHitCollection>();

  for (auto& pfrechit : *out) {
    navigator_->associateNeighbours(pfrechit, out, refProd);
  }

  iEvent.put(std::move(out), "");
  iEvent.put(std::move(cleaned), "Cleaned");
}

void PFRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
