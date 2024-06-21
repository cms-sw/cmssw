#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <vector>

class PFClusterTimeSelector : public edm::stream::EDProducer<> {
public:
  explicit PFClusterTimeSelector(const edm::ParameterSet&);
  ~PFClusterTimeSelector() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  struct CutInfo {
    double depth;
    double minE;
    double maxE;
    double minTime;
    double maxTime;
    bool endcap;
  };

  // ----------access to event data
  edm::EDGetTokenT<reco::PFClusterCollection> clusters_;
  std::vector<CutInfo> cutInfo_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFClusterTimeSelector);

void PFClusterTimeSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", {"particleFlowClusterECALWithTimeUncorrected"});
  {
    std::vector<edm::ParameterSet> vpset;
    vpset.reserve(10);
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 0.0);
      pset.addParameter<double>("maxEnergy", 1.0);
      pset.addParameter<bool>("endcap", false);
      pset.addParameter<double>("minTime", -12.);
      pset.addParameter<double>("maxTime", 12.);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 0.0);
      pset.addParameter<double>("maxEnergy", 1.0);
      pset.addParameter<bool>("endcap", true);
      pset.addParameter<double>("minTime", -31.5);
      pset.addParameter<double>("maxTime", 31.5);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 1.0);
      pset.addParameter<double>("maxEnergy", 2.0);
      pset.addParameter<bool>("endcap", false);
      pset.addParameter<double>("minTime", -6.);
      pset.addParameter<double>("maxTime", 6.);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 1.0);
      pset.addParameter<double>("maxEnergy", 2.0);
      pset.addParameter<bool>("endcap", true);
      pset.addParameter<double>("minTime", -20.5);
      pset.addParameter<double>("maxTime", 20.5);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 2.0);
      pset.addParameter<double>("maxEnergy", 5.0);
      pset.addParameter<bool>("endcap", false);
      pset.addParameter<double>("minTime", -4.);
      pset.addParameter<double>("maxTime", 4.);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 2.0);
      pset.addParameter<double>("maxEnergy", 5.0);
      pset.addParameter<bool>("endcap", true);
      pset.addParameter<double>("minTime", -12.);
      pset.addParameter<double>("maxTime", 12.);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 5.0);
      pset.addParameter<double>("maxEnergy", 20.0);
      pset.addParameter<bool>("endcap", false);
      pset.addParameter<double>("minTime", -4.);
      pset.addParameter<double>("maxTime", 4.);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 5.0);
      pset.addParameter<double>("maxEnergy", 20.0);
      pset.addParameter<bool>("endcap", true);
      pset.addParameter<double>("minTime", -5.);
      pset.addParameter<double>("maxTime", 5.);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 20.0);
      pset.addParameter<double>("maxEnergy", 1e24);
      pset.addParameter<bool>("endcap", false);
      pset.addParameter<double>("minTime", -4.);
      pset.addParameter<double>("maxTime", 4.);
      vpset.emplace_back(pset);
    }
    {
      edm::ParameterSet pset;
      pset.addParameter<double>("depth", 1.0);
      pset.addParameter<double>("minEnergy", 20.0);
      pset.addParameter<double>("maxEnergy", 1e24);
      pset.addParameter<bool>("endcap", true);
      pset.addParameter<double>("minTime", -5.);
      pset.addParameter<double>("maxTime", 5.);
      vpset.emplace_back(pset);
    }
    edm::ParameterSetDescription psd;
    psd.add<double>("depth", 1.0);
    psd.add<double>("minEnergy", 0.0);
    psd.add<double>("maxEnergy", 1e24);
    psd.add<bool>("endcap", false);
    psd.add<double>("minTime", -50.);
    psd.add<double>("maxTime", 50.);
    desc.addVPSet("cuts", psd, vpset);
  }
  descriptions.add("particleFlowClusterECALTimeSelected", desc);
}

using namespace std;
using namespace edm;

PFClusterTimeSelector::PFClusterTimeSelector(const edm::ParameterSet& iConfig)
    : clusters_(consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  std::vector<edm::ParameterSet> cuts = iConfig.getParameter<std::vector<edm::ParameterSet> >("cuts");
  for (const auto& cut : cuts) {
    CutInfo info;
    info.depth = cut.getParameter<double>("depth");
    info.minE = cut.getParameter<double>("minEnergy");
    info.maxE = cut.getParameter<double>("maxEnergy");
    info.minTime = cut.getParameter<double>("minTime");
    info.maxTime = cut.getParameter<double>("maxTime");
    info.endcap = cut.getParameter<bool>("endcap");
    cutInfo_.push_back(info);
  }

  produces<reco::PFClusterCollection>();
  produces<reco::PFClusterCollection>("OOT");
}

void PFClusterTimeSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::PFClusterCollection> clusters;
  iEvent.getByToken(clusters_, clusters);
  auto out = std::make_unique<reco::PFClusterCollection>();
  auto outOOT = std::make_unique<reco::PFClusterCollection>();

  for (const auto& cluster : *clusters) {
    const double energy = cluster.energy();
    const double time = cluster.time();
    const double depth = cluster.depth();
    const PFLayer::Layer layer = cluster.layer();
    for (const auto& info : cutInfo_) {
      if (energy < info.minE || energy > info.maxE)
        continue;
      if (depth < 0.9 * info.depth || depth > 1.1 * info.depth)
        continue;
      if ((info.endcap &&
           (layer == PFLayer::ECAL_BARREL || layer == PFLayer::HCAL_BARREL1 || layer == PFLayer::HCAL_BARREL2)) ||
          (((!info.endcap) && (layer == PFLayer::ECAL_ENDCAP || layer == PFLayer::HCAL_ENDCAP))))
        continue;

      if (time > info.minTime && time < info.maxTime)
        out->push_back(cluster);
      else
        outOOT->push_back(cluster);

      break;
    }
  }

  iEvent.put(std::move(out));
  iEvent.put(std::move(outOOT), "OOT");
}

PFClusterTimeSelector::~PFClusterTimeSelector() = default;

// ------------ method called once each job just before starting event loop  ------------
void PFClusterTimeSelector::beginRun(const edm::Run& run, const EventSetup& es) {}
