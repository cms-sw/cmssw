// -*- C++ -*-
//
// Package:    L1Trigger/L1CaloTrigger
// Class:      Phase2L1CaloPFClusterEmulator
//
/**\class Phase2L1CaloPFClusterEmulator Phase2L1CaloPFClusterEmulator.cc L1Trigger/L1CaloTrigger/plugins/Phase2L1CaloPFClusterEmulator.cc

 Description: Creates 3x3 PF clusters from GCTintTowers to be sent to correlator. Follows firmware logic, creates 8 clusters per (2+17+2)x(2+4+2).

 Implementation: To be run together with Phase2L1CaloEGammaEmulator.

*/
//
// Original Author:  Pallabi Das
//         Created:  Wed, 21 Sep 2022 14:54:20 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloPFCluster.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"

#include <ap_int.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloPFClusterEmulator.h"

//
// class declaration
//

class Phase2L1CaloPFClusterEmulator : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1CaloPFClusterEmulator(const edm::ParameterSet&);
  ~Phase2L1CaloPFClusterEmulator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<l1tp2::CaloTowerCollection> caloTowerToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Phase2L1CaloPFClusterEmulator::Phase2L1CaloPFClusterEmulator(const edm::ParameterSet& iConfig)
    : caloTowerToken_(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("gctFullTowers"))) {
  produces<l1tp2::CaloPFClusterCollection>("GCTPFCluster");
}

Phase2L1CaloPFClusterEmulator::~Phase2L1CaloPFClusterEmulator() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1CaloPFClusterEmulator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<l1tp2::CaloPFClusterCollection> pfclusterCands(make_unique<l1tp2::CaloPFClusterCollection>());

  edm::Handle<std::vector<l1tp2::CaloTower>> caloTowerCollection;
  if (!iEvent.getByToken(caloTowerToken_, caloTowerCollection))
    cms::Exception("Phase2L1CaloPFClusterEmulator") << "Failed to get towers from caloTowerCollection!";

  iEvent.getByToken(caloTowerToken_, caloTowerCollection);
  float GCTintTowers[nTowerEta][nTowerPhi];
  float realEta[nTowerEta][nTowerPhi];
  float realPhi[nTowerEta][nTowerPhi];
  for (const l1tp2::CaloTower& i : *caloTowerCollection) {
    int ieta = i.towerIEta();
    int iphi = i.towerIPhi();
    GCTintTowers[ieta][iphi] = i.ecalTowerEt();
    realEta[ieta][iphi] = i.towerEta();
    realPhi[ieta][iphi] = i.towerPhi();
  }

  float regions[nSLR][nTowerEtaSLR][nTowerPhiSLR];

  for (int ieta = 0; ieta < nTowerEtaSLR; ieta++) {
    for (int iphi = 0; iphi < nTowerPhiSLR; iphi++) {
      for (int k = 0; k < nSLR; k++) {
        regions[k][ieta][iphi] = 0.;
      }
    }
  }

  //Assign ETs to 36 21x8 regions

  for (int ieta = 0; ieta < nTowerEtaSLR; ieta++) {
    for (int iphi = 0; iphi < nTowerPhiSLR; iphi++) {
      if (ieta > 1) {
        if (iphi > 1)
          regions[0][ieta][iphi] = GCTintTowers[ieta - 2][iphi - 2];
        for (int k = 1; k < 17; k++) {
          regions[k * 2][ieta][iphi] = GCTintTowers[ieta - 2][iphi + k * 4 - 2];
        }
        if (iphi < 6)
          regions[34][ieta][iphi] = GCTintTowers[ieta - 2][iphi + 66];
      }
      if (ieta < 19) {
        if (iphi > 1)
          regions[1][ieta][iphi] = GCTintTowers[ieta + 15][iphi - 2];
        for (int k = 1; k < 17; k++) {
          regions[k * 2 + 1][ieta][iphi] = GCTintTowers[ieta + 15][iphi + k * 4 - 2];
        }
        if (iphi < 6)
          regions[35][ieta][iphi] = GCTintTowers[ieta + 15][iphi + 66];
      }
    }
  }

  float temporary[nTowerEtaSLR][nTowerPhiSLR];
  int etaoffset = 0;
  int phioffset = 0;

  //Use same code from firmware for finding clusters
  for (int k = 0; k < nSLR; k++) {
    for (int ieta = 0; ieta < nTowerEtaSLR; ieta++) {
      for (int iphi = 0; iphi < nTowerPhiSLR; iphi++) {
        temporary[ieta][iphi] = regions[k][ieta][iphi];
      }
    }
    if (k % 2 == 0)
      etaoffset = 0;
    else
      etaoffset = 17;
    if (k > 1 && k % 2 == 0)
      phioffset = phioffset + 4;

    gctpf::GCTPfcluster_t tempPfclusters;
    tempPfclusters = gctpf::pfcluster(temporary, etaoffset, phioffset);

    for (int i = 0; i < nPFClusterSLR; i++) {
      int gcteta = tempPfclusters.GCTpfclusters[i].eta;
      int gctphi = tempPfclusters.GCTpfclusters[i].phi;
      float towereta = realEta[gcteta][gctphi];
      float towerphi = realPhi[gcteta][gctphi];
      l1tp2::CaloPFCluster l1CaloPFCluster;
      l1CaloPFCluster.setClusterEt(tempPfclusters.GCTpfclusters[i].et);
      l1CaloPFCluster.setClusterIEta(gcteta);
      l1CaloPFCluster.setClusterIPhi(gctphi);
      l1CaloPFCluster.setClusterEta(towereta);
      l1CaloPFCluster.setClusterPhi(towerphi);
      pfclusterCands->push_back(l1CaloPFCluster);
    }
  }

  iEvent.put(std::move(pfclusterCands), "GCTPFCluster");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase2L1CaloPFClusterEmulator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gctFullTowers", edm::InputTag("l1tPhase2L1CaloEGammaEmulator", "GCTFullTowers"));
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1CaloPFClusterEmulator);
