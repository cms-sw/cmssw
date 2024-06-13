#include "RecoParticleFlow/PFClusterProducer/test/PFClusterAnalyzer.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace std;
using namespace edm;
using namespace reco;

PFClusterAnalyzer::PFClusterAnalyzer(const edm::ParameterSet& iConfig)
    : inputTokenPFClusters_(consumes<reco::PFClusterCollection>(iConfig.getParameter<InputTag>("PFClusters"))),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      printBlocks_(iConfig.getUntrackedParameter<bool>("printBlocks", false)) {
  LogDebug("PFClusterAnalyzer") << " input collection : " << iConfig.getParameter<InputTag>("PFClusters");
}

void PFClusterAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& es) {}

void PFClusterAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup) {
  LogDebug("PFClusterAnalyzer") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;

  // get PFClusters

  Handle<PFClusterCollection> pfClusters;
  fetchCandidateCollection(pfClusters, inputTokenPFClusters_, iEvent);

  // get PFClusters for isolation

  for (unsigned i = 0; i < pfClusters->size(); i++) {
    const reco::PFCluster& cluster = (*pfClusters)[i];

    if (verbose_) {
      cout << "PFCluster " << endl;
      cout << cluster << endl;
      cout << "CaloCluster " << endl;

      const CaloCluster* caloc = dynamic_cast<const CaloCluster*>(&cluster);
      assert(caloc);
      cout << *caloc << endl;
      cout << endl;
    }
  }

  LogDebug("PFClusterAnalyzer") << "STOP event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;
}

void PFClusterAnalyzer::fetchCandidateCollection(Handle<reco::PFClusterCollection>& c,
                                                 const edm::EDGetTokenT<reco::PFClusterCollection>& token,
                                                 const Event& iEvent) const {
  c = iEvent.getHandle(token);

  if (!c.isValid()) {
    ostringstream err;
    err << " cannot get PFClusters " << endl;
    LogError("PFClusters") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }
}

DEFINE_FWK_MODULE(PFClusterAnalyzer);
