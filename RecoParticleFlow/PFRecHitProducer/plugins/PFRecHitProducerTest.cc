#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>

class PFRecHitProducerTest : public DQMEDAnalyzer {
public:
  PFRecHitProducerTest(edm::ParameterSet const& conf);
  ~PFRecHitProducerTest() override = default;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override {};
  //static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit>> recHitsToken;
  edm::EDGetTokenT<reco::PFRecHitCollection> pfRecHitsTokenCPU;
  edm::EDGetTokenT<PFRecHitHostCollection> pfRecHitsTokenAlpaka;
};

PFRecHitProducerTest::PFRecHitProducerTest(const edm::ParameterSet& conf)
    : recHitsToken(
          consumes<edm::SortedCollection<HBHERecHit>>(conf.getUntrackedParameter<edm::InputTag>("recHitsSourceCPU"))),
      pfRecHitsTokenCPU(
          consumes<reco::PFRecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceCPU"))),
      pfRecHitsTokenAlpaka(
          consumes<PFRecHitHostCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceAlpaka")))
     {}

void PFRecHitProducerTest::analyze(edm::Event const& event, edm::EventSetup const& c) {
  static int cnt = 0;
  if (cnt++ >= 1)
    return;

  // Rec Hits
  //edm::Handle<edm::SortedCollection<HBHERecHit>> recHits;
  //event.getByToken(recHitsToken, recHits);
  //printf("Found %zd recHits\n", recHits->size());
  //fprintf(stderr, "Found %zd recHits\n", recHits->size());
  //for (size_t i = 0; i < recHits->size(); i++)
  //  printf("recHit %4lu %u\n", i, recHits->operator[](i).id().rawId());

  // PF Rec Hits
  // paste <(grep "^CPU" validation.log | sort -nk3) <(grep "^GPU" validation.log | sort -nk3) | awk '$3!=$13 || $4!=$14 || $5!=$15 || $6!=$16 || $9!=$19 {print}' | head
  edm::Handle<reco::PFRecHitCollection> pfRecHitsCPU;
  edm::Handle<PFRecHitHostCollection> pfRecHitsAlpakaSoA;
  event.getByToken(pfRecHitsTokenCPU, pfRecHitsCPU);
  event.getByToken(pfRecHitsTokenAlpaka, pfRecHitsAlpakaSoA);
  auto& pfRecHitsAlpaka = pfRecHitsAlpakaSoA->view();

  fprintf(stdout, "Found %zd/%d pfRecHits with CPU/Alpaka\n", pfRecHitsCPU->size(), pfRecHitsAlpaka.size());
  fprintf(stderr, "Found %zd/%d pfRecHits with CPU/Alpaka\n", pfRecHitsCPU->size(), pfRecHitsAlpaka.size());
  for (size_t i = 0; i < pfRecHitsCPU->size(); i++)
    printf("CPU %4lu %u %d %d %u : %f %f (%f,%f,%f)\n",
           i,
           pfRecHitsCPU->at(i).detId(),
           pfRecHitsCPU->at(i).depth(),
           pfRecHitsCPU->at(i).layer(),
           pfRecHitsCPU->at(i).neighbours().size(),
           pfRecHitsCPU->at(i).time(),
           pfRecHitsCPU->at(i).energy(),
           0.,  //pfRecHitsCPU->at(i).position().x(),
           0.,  //pfRecHitsCPU->at(i).position().y(),
           0.   //pfRecHitsCPU->at(i).position().z()
    );
  for (size_t i = 0; i < pfRecHitsAlpaka.size(); i++)
    printf("Alpaka %4lu %u %d %d %u : %f %f (%f,%f,%f)\n",
           i,
           pfRecHitsAlpaka[i].detId(),
           pfRecHitsAlpaka[i].depth(),
           pfRecHitsAlpaka[i].layer(),
           -1,//pfRecHitsAlpaka[i].neighbours().size(),
           pfRecHitsAlpaka[i].time(),
           pfRecHitsAlpaka[i].energy(),
           0.,  //pfRecHitsGPU->at(i).position().x(),
           0.,  //pfRecHitsGPU->at(i).position().y(),
           0.   //pfRecHitsGPU->at(i).position().z()
    );
}

// void PFRecHitProducerTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   edm::ParameterSetDescription desc;
//   //desc.setUnknown();
//   desc.add<edm::InputTag>("pfClusterToken_ref", edm::InputTag("particleFlowClusterHBHE"));
//   desc.add<edm::InputTag>("pfClusterToken_target", edm::InputTag("particleFlowClusterHBHEonGPU"));
//   desc.addUntracked<std::string>("pfCaloGPUCompDir", "pfClusterHBHEGPUv");
//   descriptions.addDefault(desc);
// }

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecHitProducerTest);
