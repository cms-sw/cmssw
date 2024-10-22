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
  std::vector<std::unique_ptr<PFRecHitCreatorBase>> creators_;
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

  std::vector<edm::ParameterSet> creators = iConfig.getParameter<std::vector<edm::ParameterSet>>("producers");
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
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription pset;
    pset.add<std::string>("name", "");
    pset.add<std::vector<int>>("hcalEnums", {});
    pset.add<edm::ParameterSetDescription>("barrel", {});
    pset.add<edm::ParameterSetDescription>("endcap", {});
    {
      edm::ParameterSetDescription pset2;
      pset2.add<std::string>("name", "");
      pset2.add<std::string>("topologySource", "");
      pset.add<edm::ParameterSetDescription>("hgcee", pset2);
      pset.add<edm::ParameterSetDescription>("hgcheb", pset2);
      pset.add<edm::ParameterSetDescription>("hgchef", pset2);
    }
    desc.add<edm::ParameterSetDescription>("navigator", pset);
  }
  {
    edm::ParameterSetDescription psd;
    psd.add<std::string>("name", "");
    psd.add<edm::InputTag>("src", {});
    {
      edm::ParameterSetDescription psd2;
      psd2.add<std::string>("name", "");
      psd2.add<std::vector<int>>("maxSeverities", {});
      psd2.add<std::vector<double>>("cleaningThresholds", {});
      psd2.add<std::vector<std::string>>("flags", {});
      psd2.add<bool>("usePFThresholdsFromDB", false);
      {
        edm::ParameterSetDescription psd3;
        psd3.add<std::vector<int>>("depth", {});
        psd3.add<std::vector<double>>("threshold", {});
        psd3.add<int>("detectorEnum", 0);
        psd2.addVPSet("cuts", psd3, {});
      }
      psd2.add<double>("thresholdSNR", 0);
      psd2.add<bool>("applySelectionsToAllCrystals", false);
      psd2.add<double>("cleaningThreshold", 0);
      psd2.add<bool>("timingCleaning", false);
      psd2.add<bool>("topologicalCleaning", false);
      psd2.add<bool>("skipTTRecoveredHits", false);
      psd2.add<double>("threshold", 0);
      psd2.add<double>("threshold_ring0", 0);
      psd2.add<double>("threshold_ring12", 0);
      psd.addVPSet("qualityTests", psd2, {});
    }
    psd.add<double>("EMDepthCorrection", 0);
    psd.add<double>("HADDepthCorrection", 0);
    psd.add<double>("thresh_HF", 0);
    psd.add<double>("ShortFibre_Cut", 0);
    psd.add<double>("LongFibre_Fraction", 0);
    psd.add<double>("LongFibre_Cut", 0);
    psd.add<double>("ShortFibre_Fraction", 0);
    psd.add<double>("HFCalib29", 0);
    psd.add<edm::InputTag>("srFlags", {});
    psd.add<std::string>("geometryInstance", "");
    desc.addVPSet("producers", psd, {});
  }
  descriptions.addWithDefaultLabel(desc);
}
