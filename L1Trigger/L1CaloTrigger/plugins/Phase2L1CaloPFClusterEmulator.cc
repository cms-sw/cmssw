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
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

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
  ~Phase2L1CaloPFClusterEmulator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<l1tp2::CaloTowerCollection> caloTowerToken_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hfToken_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoderTag_;
};

//
// constructors and destructor
//
Phase2L1CaloPFClusterEmulator::Phase2L1CaloPFClusterEmulator(const edm::ParameterSet& iConfig)
    : caloTowerToken_(consumes<l1tp2::CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("gctFullTowers"))),
      hfToken_(consumes<HcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalDigis"))),
      decoderTag_(esConsumes<CaloTPGTranscoder, CaloTPGRecord>(edm::ESInputTag("", ""))) {
  produces<l1tp2::CaloPFClusterCollection>("GCTPFCluster");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1CaloPFClusterEmulator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<l1tp2::CaloPFClusterCollection> pfclusterCands(make_unique<l1tp2::CaloPFClusterCollection>());

  edm::Handle<std::vector<l1tp2::CaloTower>> caloTowerCollection;
  iEvent.getByToken(caloTowerToken_, caloTowerCollection);
  if (!caloTowerCollection.isValid())
    cms::Exception("Phase2L1CaloPFClusterEmulator") << "Failed to get towers from caloTowerCollection!";

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

    gctpf::PFcluster_t tempPfclusters;
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

  edm::Handle<HcalTrigPrimDigiCollection> hfHandle;
  if (!iEvent.getByToken(hfToken_, hfHandle))
    edm::LogError("Phase2L1CaloJetEmulator") << "Failed to get HcalTrigPrimDigi for HF!";
  iEvent.getByToken(hfToken_, hfHandle);

  float hfTowers[2 * nHfEta][nHfPhi];  // split 12 -> 24
  float hfEta[nHfEta][nHfPhi];
  float hfPhi[nHfEta][nHfPhi];
  for (int iphi = 0; iphi < nHfPhi; iphi++) {
    for (int ieta = 0; ieta < nHfEta; ieta++) {
      hfTowers[2 * ieta][iphi] = 0;
      hfTowers[2 * ieta + 1][iphi] = 0;
      int temp;
      if (ieta < nHfEta / 2)
        temp = ieta - l1t::CaloTools::kHFEnd;
      else
        temp = ieta - nHfEta / 2 + l1t::CaloTools::kHFBegin + 1;
      hfEta[ieta][iphi] = l1t::CaloTools::towerEta(temp);
      hfPhi[ieta][iphi] = l1t::CaloTools::towerPhi(temp, iphi + 1);
    }
  }

  float regionsHF[nHfRegions][nHfEta]
                 [nHfPhi /
                  6];  // 6 regions each 3 sectors (1 sector = 20 deg in phi), 12x12->24x12 unique towers, no overlap in phi

  for (int ieta = 0; ieta < nHfEta; ieta++) {
    for (int iphi = 0; iphi < nHfPhi / 6; iphi++) {
      for (int k = 0; k < nHfRegions; k++) {
        regionsHF[k][ieta][iphi] = 0.;
      }
    }
  }

  // Assign ET to 12 24x12 regions
  const auto& decoder = iSetup.getData(decoderTag_);
  for (const auto& hit : *hfHandle.product()) {
    double et = decoder.hcaletValue(hit.id(), hit.t0());
    int ieta = 0;
    if (abs(hit.id().ieta()) < l1t::CaloTools::kHFBegin)
      continue;
    if (abs(hit.id().ieta()) > l1t::CaloTools::kHFEnd)
      continue;
    if (hit.id().ieta() <= -(l1t::CaloTools::kHFBegin + 1)) {
      ieta = hit.id().ieta() + l1t::CaloTools::kHFEnd;
    } else if (hit.id().ieta() >= (l1t::CaloTools::kHFBegin + 1)) {
      ieta = nHfEta / 2 + (hit.id().ieta() - (l1t::CaloTools::kHFBegin + 1));
    }
    int iphi = hit.id().iphi() - 1;  // HF phi runs between 1-72
    // split tower energy
    hfTowers[2 * ieta][iphi] = et / 2;
    hfTowers[2 * ieta + 1][iphi] = et / 2;
    if ((ieta < 2 || ieta >= nHfEta - 2) && iphi % 4 == 2) {
      hfTowers[2 * ieta][iphi] = et / 8;
      hfTowers[2 * ieta + 1][iphi] = et / 8;
      hfTowers[2 * ieta][iphi + 1] = et / 8;
      hfTowers[2 * ieta + 1][iphi + 1] = et / 8;
      hfTowers[2 * ieta][iphi + 2] = et / 8;
      hfTowers[2 * ieta + 1][iphi + 2] = et / 8;
      hfTowers[2 * ieta][iphi + 3] = et / 8;
      hfTowers[2 * ieta + 1][iphi + 3] = et / 8;
    } else if ((ieta >= 2 && ieta < nHfEta - 2) && iphi % 2 == 0) {
      hfTowers[2 * ieta][iphi] = et / 4;
      hfTowers[2 * ieta + 1][iphi] = et / 4;
      hfTowers[2 * ieta][iphi + 1] = et / 4;
      hfTowers[2 * ieta + 1][iphi + 1] = et / 4;
    }
  }

  for (int ieta = 0; ieta < 2 * nHfEta; ieta++) {
    for (int iphi = 0; iphi < nHfPhi / 6; iphi++) {
      if (ieta < nHfEta) {
        regionsHF[0][ieta][0] = hfTowers[ieta][70];
        regionsHF[0][ieta][1] = hfTowers[ieta][71];
        for (int k = 0; k < nHfRegions; k += 2) {
          if (k != 0 || iphi > 1)
            regionsHF[k][ieta][iphi] = hfTowers[ieta][iphi + k * (nHfRegions / 2) - 2];
        }
      } else {
        regionsHF[1][ieta - nHfEta][0] = hfTowers[ieta][70];
        regionsHF[1][ieta - nHfEta][1] = hfTowers[ieta][71];
        for (int k = 1; k < nHfRegions; k += 2) {
          if (k != 1 || iphi > 1)
            regionsHF[k][ieta - nHfEta][iphi] = hfTowers[ieta][iphi + (k - 1) * (nHfRegions / 2) - 2];
        }
      }
    }
  }

  float temporaryHF[nHfEta][nHfPhi / 6];
  int etaoffsetHF = 0;
  int phioffsetHF = -2;

  for (int k = 0; k < nHfRegions; k++) {
    for (int ieta = 0; ieta < nHfEta; ieta++) {
      for (int iphi = 0; iphi < nHfPhi / 6; iphi++) {
        temporaryHF[ieta][iphi] = regionsHF[k][ieta][iphi];
      }
    }
    if (k % 2 == 0)
      etaoffsetHF = 0;
    else
      etaoffsetHF = nHfEta;
    if (k > 1 && k % 2 == 0)
      phioffsetHF = phioffsetHF + nHfPhi / 6;

    gctpf::PFcluster_t tempPfclustersHF;
    tempPfclustersHF = gctpf::pfclusterHF(temporaryHF, etaoffsetHF, phioffsetHF);

    for (int i = 0; i < nPFClusterSLR; i++) {
      int hfeta = tempPfclustersHF.GCTpfclusters[i].eta / 2;  // turn back 24 -> 12
      int hfphi = tempPfclustersHF.GCTpfclusters[i].phi;
      float towereta = hfEta[hfeta][hfphi];
      float towerphi = hfPhi[hfeta][hfphi];
      l1tp2::CaloPFCluster l1CaloPFCluster;
      l1CaloPFCluster.setClusterEt(tempPfclustersHF.GCTpfclusters[i].et);
      l1CaloPFCluster.setClusterIEta(hfeta);
      l1CaloPFCluster.setClusterIPhi(hfphi);
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
  desc.add<edm::InputTag>("hcalDigis", edm::InputTag("simHcalTriggerPrimitiveDigis"));
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1CaloPFClusterEmulator);
