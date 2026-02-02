/* AS */

// system include files
#include <ap_int.h>
#include <array>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloToCorrelatorTMI18.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedCaloToCorrelatorTMI18.h"

class Phase2L1CaloToCorrelatorTMI18 : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1CaloToCorrelatorTMI18(const edm::ParameterSet&);
  ~Phase2L1CaloToCorrelatorTMI18() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<l1tp2::GCTEmDigiClusterCollection> gctEmDigiClustersSrc_;
  edm::EDGetTokenT<l1tp2::GCTHadDigiClusterCollection> gctHadDigiClustersSrc_;
};

Phase2L1CaloToCorrelatorTMI18::Phase2L1CaloToCorrelatorTMI18(const edm::ParameterSet& cfg)
    : gctEmDigiClustersSrc_(
          consumes<l1tp2::GCTEmDigiClusterCollection>(cfg.getParameter<edm::InputTag>("gctEmDigiClusters"))),
      gctHadDigiClustersSrc_(
          consumes<l1tp2::GCTHadDigiClusterCollection>(cfg.getParameter<edm::InputTag>("gctHadDigiClusters"))) {
  produces<l1tp2::DigitizedCaloToCorrelatorCollectionTMI18>("DigitizedCaloToCorrelatorTMI18");
}

void Phase2L1CaloToCorrelatorTMI18::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;
  std::unique_ptr<l1tp2::DigitizedCaloToCorrelatorCollectionTMI18> caloCandsTMI18(
      std::make_unique<l1tp2::DigitizedCaloToCorrelatorCollectionTMI18>());

  int EM_SLR1_POS_OFFSET = 1;
  int EM_SLR1_NEG_OFFSET = 17;
  int PF_SLR1_POS_OFFSET = 33;
  int PF_SLR1_NEG_OFFSET = 57;
  int EM_SLR3_POS_OFFSET = 82;
  int EM_SLR3_NEG_OFFSET = 98;
  int PF_SLR3_POS_OFFSET = 114;
  int PF_SLR3_NEG_OFFSET = 138;
  int NUM_EM_WORDS = 16;
  int NUM_PF_WORDS = 24;

  int cntr03pos = 0;
  int cntr03neg = 0;
  int cntr01pos = 0;
  int cntr01neg = 0;

  int cntr13pos = 0;
  int cntr13neg = 0;
  int cntr11pos = 0;
  int cntr11neg = 0;

  int cntr23pos = 0;
  int cntr23neg = 0;
  int cntr21pos = 0;
  int cntr21neg = 0;

  ap_uint<64> mydata = 0;
  ap_uint<64> dataToCL1Card0[162] = {0};
  ap_uint<64> dataToCL1Card1[162] = {0};
  ap_uint<64> dataToCL1Card2[162] = {0};
  l1tp2::GCTDigiClusterLink clusterCollCard0(162);
  l1tp2::GCTDigiClusterLink clusterCollCard1(162);
  l1tp2::GCTDigiClusterLink clusterCollCard2(162);

  // SLR1 and SLR3 both send 24 PFclusters each from +ve and -ve eta, total 48 words: 4x12(x64b)

  edm::Handle<l1tp2::GCTHadDigiClusterCollection> gctHadDigiClusters;
  if (evt.getByToken(gctHadDigiClustersSrc_, gctHadDigiClusters)) {
    for (int iLink = 0; iLink < 12; iLink++) {
      // in order:
      // GCT1 SLR3 +ve, GCT1 SLR3 -ve, GCT1 SLR1 +ve, GCT1 SLR1 -ve
      // GCT2 SLR3 +ve, GCT2 SLR3 -ve, GCT2 SLR1 +ve, GCT2 SLR1 -ve
      // GCT3 SLR3 +ve, GCT3 SLR3 -ve, GCT3 SLR1 +ve, GCT3 SLR1 -ve
      int iGCT = (iLink / 4);
      // 4 RCT regions per GCT
      int iRCT = iLink - iGCT * 4;
      // Eta: positive or negative depends on the link. If iLink is even, it is in positive eta
      bool isNegativeEta = (iRCT % 2 == 1);
      // SLR alternates every two links
      bool isSLR3 = ((iLink % 4) < 2);
      bool isSLR1 = !isSLR3;
      for (const auto& cluster : (*gctHadDigiClusters).at(iLink)) {
        mydata = cluster.data();

        if ((iGCT == 0) && isSLR1 && !isNegativeEta && (cntr01pos < NUM_PF_WORDS)) {
          dataToCL1Card0[PF_SLR1_POS_OFFSET + cntr01pos] = mydata;
          clusterCollCard0[PF_SLR1_POS_OFFSET + cntr01pos] = cluster;
          cntr01pos++;
        }
        if ((iGCT == 1) && isSLR1 && !isNegativeEta && (cntr11pos < NUM_PF_WORDS)) {
          dataToCL1Card1[PF_SLR1_POS_OFFSET + cntr11pos] = mydata;
          clusterCollCard1[PF_SLR1_POS_OFFSET + cntr11pos] = cluster;
          cntr11pos++;
        }
        if ((iGCT == 2) && isSLR1 && !isNegativeEta && (cntr21pos < NUM_PF_WORDS)) {
          dataToCL1Card2[PF_SLR1_POS_OFFSET + cntr21pos] = mydata;
          clusterCollCard2[PF_SLR1_POS_OFFSET + cntr21pos] = cluster;
          cntr21pos++;
        }
        if ((iGCT == 0) && isSLR1 && isNegativeEta && (cntr01neg < NUM_PF_WORDS)) {
          dataToCL1Card0[PF_SLR1_NEG_OFFSET + cntr01neg] = mydata;
          clusterCollCard0[PF_SLR1_NEG_OFFSET + cntr01neg] = cluster;
          cntr01neg++;
        }
        if ((iGCT == 1) && isSLR1 && isNegativeEta && (cntr11neg < NUM_PF_WORDS)) {
          dataToCL1Card1[PF_SLR1_NEG_OFFSET + cntr11neg] = mydata;
          clusterCollCard1[PF_SLR1_NEG_OFFSET + cntr11neg] = cluster;
          cntr11neg++;
        }
        if ((iGCT == 2) && isSLR1 && isNegativeEta && (cntr21neg < NUM_PF_WORDS)) {
          dataToCL1Card2[PF_SLR1_NEG_OFFSET + cntr21neg] = mydata;
          clusterCollCard2[PF_SLR1_NEG_OFFSET + cntr21neg] = cluster;
          cntr21neg++;
        }
        if ((iGCT == 0) && isSLR3 && !isNegativeEta && (cntr03pos < NUM_PF_WORDS)) {
          dataToCL1Card0[PF_SLR3_POS_OFFSET + cntr03pos] = mydata;
          clusterCollCard0[PF_SLR3_POS_OFFSET + cntr03pos] = cluster;
          cntr03pos++;
        }
        if ((iGCT == 1) && isSLR3 && !isNegativeEta && (cntr13pos < NUM_PF_WORDS)) {
          dataToCL1Card1[PF_SLR3_POS_OFFSET + cntr13pos] = mydata;
          clusterCollCard1[PF_SLR3_POS_OFFSET + cntr13pos] = cluster;
          cntr13pos++;
        }
        if ((iGCT == 2) && isSLR3 && !isNegativeEta && (cntr23pos < NUM_PF_WORDS)) {
          dataToCL1Card2[PF_SLR3_POS_OFFSET + cntr23pos] = mydata;
          clusterCollCard2[PF_SLR3_POS_OFFSET + cntr23pos] = cluster;
          cntr23pos++;
        }
        if ((iGCT == 0) && isSLR3 && isNegativeEta && (cntr03neg < NUM_PF_WORDS)) {
          dataToCL1Card0[PF_SLR3_NEG_OFFSET + cntr03neg] = mydata;
          clusterCollCard0[PF_SLR3_NEG_OFFSET + cntr03neg] = cluster;
          cntr03neg++;
        }
        if ((iGCT == 1) && isSLR3 && isNegativeEta && (cntr13neg < NUM_PF_WORDS)) {
          dataToCL1Card1[PF_SLR3_NEG_OFFSET + cntr13neg] = mydata;
          clusterCollCard1[PF_SLR3_NEG_OFFSET + cntr13neg] = cluster;
          cntr13neg++;
        }
        if ((iGCT == 2) && isSLR3 && isNegativeEta && (cntr23neg < NUM_PF_WORDS)) {
          dataToCL1Card2[PF_SLR3_NEG_OFFSET + cntr23neg] = mydata;
          clusterCollCard2[PF_SLR3_NEG_OFFSET + cntr23neg] = cluster;
          cntr23neg++;
        }
      }
    }
  }

  cntr03pos = 0;
  cntr03neg = 0;
  cntr01pos = 0;
  cntr01neg = 0;

  cntr13pos = 0;
  cntr13neg = 0;
  cntr11pos = 0;
  cntr11neg = 0;

  cntr23pos = 0;
  cntr23neg = 0;
  cntr21pos = 0;
  cntr21neg = 0;

  edm::Handle<l1tp2::GCTEmDigiClusterCollection> gctEmDigiClusters;
  if (evt.getByToken(gctEmDigiClustersSrc_, gctEmDigiClusters)) {
    for (int iLink = 0; iLink < 12; iLink++) {
      // in order:
      // GCT1 SLR3 +ve, GCT1 SLR3 -ve, GCT1 SLR1 +ve, GCT1 SLR1 -ve
      // GCT2 SLR3 +ve, GCT2 SLR3 -ve, GCT2 SLR1 +ve, GCT2 SLR1 -ve
      // GCT3 SLR3 +ve, GCT3 SLR3 -ve, GCT3 SLR1 +ve, GCT3 SLR1 -ve
      int iGCT = (iLink / 4);
      // 4 RCT regions per GCT
      int iRCT = iLink - iGCT * 4;
      // Eta: positive or negative depends on the link. If iLink is even, it is in positive eta
      bool isNegativeEta = (iRCT % 2 == 1);
      // SLR alternates every two links
      bool isSLR3 = ((iLink % 4) < 2);
      bool isSLR1 = !isSLR3;
      for (const auto& cluster : (*gctEmDigiClusters).at(iLink)) {
        mydata = cluster.data();

        if ((iGCT == 0) && isSLR1 && !isNegativeEta && (cntr01pos < NUM_EM_WORDS)) {
          dataToCL1Card0[EM_SLR1_POS_OFFSET + cntr01pos] = mydata;
          clusterCollCard0[EM_SLR1_POS_OFFSET + cntr01pos] = cluster;
          cntr01pos++;
        }
        if ((iGCT == 1) && isSLR1 && !isNegativeEta && (cntr11pos < NUM_EM_WORDS)) {
          dataToCL1Card1[EM_SLR1_POS_OFFSET + cntr11pos] = mydata;
          clusterCollCard1[EM_SLR1_POS_OFFSET + cntr11pos] = cluster;
          cntr11pos++;
        }
        if ((iGCT == 2) && isSLR1 && !isNegativeEta && (cntr21pos < NUM_EM_WORDS)) {
          dataToCL1Card2[EM_SLR1_POS_OFFSET + cntr21pos] = mydata;
          clusterCollCard2[EM_SLR1_POS_OFFSET + cntr21pos] = cluster;
          cntr21pos++;
        }
        if ((iGCT == 0) && isSLR1 && isNegativeEta && (cntr01neg < NUM_EM_WORDS)) {
          dataToCL1Card0[EM_SLR1_NEG_OFFSET + cntr01neg] = mydata;
          clusterCollCard0[EM_SLR1_NEG_OFFSET + cntr01neg] = cluster;
          cntr01neg++;
        }
        if ((iGCT == 1) && isSLR1 && isNegativeEta && (cntr11neg < NUM_EM_WORDS)) {
          dataToCL1Card1[EM_SLR1_NEG_OFFSET + cntr11neg] = mydata;
          clusterCollCard1[EM_SLR1_NEG_OFFSET + cntr11neg] = cluster;
          cntr11neg++;
        }
        if ((iGCT == 2) && isSLR1 && isNegativeEta && (cntr21neg < NUM_EM_WORDS)) {
          dataToCL1Card2[EM_SLR1_NEG_OFFSET + cntr21neg] = mydata;
          clusterCollCard2[EM_SLR1_NEG_OFFSET + cntr21neg] = cluster;
          cntr21neg++;
        }
        if ((iGCT == 0) && isSLR3 && !isNegativeEta && (cntr03pos < NUM_EM_WORDS)) {
          dataToCL1Card0[EM_SLR3_POS_OFFSET + cntr03pos] = mydata;
          clusterCollCard0[EM_SLR3_POS_OFFSET + cntr03pos] = cluster;
          cntr03pos++;
        }
        if ((iGCT == 1) && isSLR3 && !isNegativeEta && (cntr13pos < NUM_EM_WORDS)) {
          dataToCL1Card1[EM_SLR3_POS_OFFSET + cntr13pos] = mydata;
          clusterCollCard1[EM_SLR3_POS_OFFSET + cntr13pos] = cluster;
          cntr13pos++;
        }
        if ((iGCT == 2) && isSLR3 && !isNegativeEta && (cntr23pos < NUM_EM_WORDS)) {
          dataToCL1Card2[EM_SLR3_POS_OFFSET + cntr23pos] = mydata;
          clusterCollCard2[EM_SLR3_POS_OFFSET + cntr23pos] = cluster;
          cntr23pos++;
        }
        if ((iGCT == 0) && isSLR3 && isNegativeEta && (cntr03neg < NUM_EM_WORDS)) {
          dataToCL1Card0[EM_SLR3_NEG_OFFSET + cntr03neg] = mydata;
          clusterCollCard0[EM_SLR3_NEG_OFFSET + cntr03neg] = cluster;
          cntr03neg++;
        }
        if ((iGCT == 1) && isSLR3 && isNegativeEta && (cntr13neg < NUM_EM_WORDS)) {
          dataToCL1Card1[EM_SLR3_NEG_OFFSET + cntr13neg] = mydata;
          clusterCollCard1[EM_SLR3_NEG_OFFSET + cntr13neg] = cluster;
          cntr13neg++;
        }
        if ((iGCT == 2) && isSLR3 && isNegativeEta && (cntr23neg < NUM_EM_WORDS)) {
          dataToCL1Card2[EM_SLR3_NEG_OFFSET + cntr23neg] = mydata;
          clusterCollCard2[EM_SLR3_NEG_OFFSET + cntr23neg] = cluster;
          cntr23neg++;
        }
      }
    }
  }

  l1tp2::DigitizedCaloToCorrelatorTMI18 l1CaloTMI18_0 =
      l1tp2::DigitizedCaloToCorrelatorTMI18(dataToCL1Card0, clusterCollCard0);
  l1tp2::DigitizedCaloToCorrelatorTMI18 l1CaloTMI18_1 =
      l1tp2::DigitizedCaloToCorrelatorTMI18(dataToCL1Card1, clusterCollCard1);
  l1tp2::DigitizedCaloToCorrelatorTMI18 l1CaloTMI18_2 =
      l1tp2::DigitizedCaloToCorrelatorTMI18(dataToCL1Card2, clusterCollCard2);

  caloCandsTMI18->push_back(l1CaloTMI18_0);
  caloCandsTMI18->push_back(l1CaloTMI18_1);
  caloCandsTMI18->push_back(l1CaloTMI18_2);
  evt.put(std::move(caloCandsTMI18), "DigitizedCaloToCorrelatorTMI18");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase2L1CaloToCorrelatorTMI18::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gctEmDigiClusters",
                          edm::InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTEmDigiClusters"));
  desc.add<edm::InputTag>("gctHadDigiClusters",
                          edm::InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTHadDigiClusters"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Phase2L1CaloToCorrelatorTMI18);
