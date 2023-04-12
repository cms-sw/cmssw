/* 
 * Description: Phase 2 RCT and GCT emulator
 */

// system include files
#include <ap_int.h>
#include <array>
#include <cmath>
// #include <cstdint>
#include <cstdlib>  // for rand
#include <iostream>
#include <fstream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

// ECAL TPs
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

// HCAL TPs
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

// Output tower collection
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelator.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedTowerCorrelator.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterGT.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "L1Trigger/L1CaloTrigger/interface/ParametricCalibration.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaEmulator.h"
#include "L1Trigger/L1CaloTrigger/interface/Phase2L1RCT.h"
#include "L1Trigger/L1CaloTrigger/interface/Phase2L1GCT.h"

// Declare the Phase2L1CaloEGammaEmulator class and its methods

class Phase2L1CaloEGammaEmulator : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1CaloEGammaEmulator(const edm::ParameterSet&);
  ~Phase2L1CaloEGammaEmulator() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<EcalEBTrigPrimDigiCollection> ecalTPEBToken_;
  edm::EDGetTokenT<edm::SortedCollection<HcalTriggerPrimitiveDigi>> hcalTPToken_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoderTag_;

  l1tp2::ParametricCalibration calib_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryTag_;
  const CaloSubdetectorGeometry* ebGeometry;
  const CaloSubdetectorGeometry* hbGeometry;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hbTopologyTag_;
  const HcalTopology* hcTopology_;
};

//////////////////////////////////////////////////////////////////////////

// Phase2L1CaloEGammaEmulator initializer, destructor, and produce methods

Phase2L1CaloEGammaEmulator::Phase2L1CaloEGammaEmulator(const edm::ParameterSet& iConfig)
    : ecalTPEBToken_(consumes<EcalEBTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalTPEB"))),
      hcalTPToken_(
          consumes<edm::SortedCollection<HcalTriggerPrimitiveDigi>>(iConfig.getParameter<edm::InputTag>("hcalTP"))),
      decoderTag_(esConsumes<CaloTPGTranscoder, CaloTPGRecord>(edm::ESInputTag("", ""))),
      calib_(iConfig.getParameter<edm::ParameterSet>("calib")),
      caloGeometryTag_(esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag("", ""))),
      hbTopologyTag_(esConsumes<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag("", ""))) {
  produces<l1tp2::CaloCrystalClusterCollection>("RCT");
  produces<l1tp2::CaloCrystalClusterCollection>("GCT");
  produces<l1tp2::CaloTowerCollection>("RCT");
  produces<l1tp2::CaloTowerCollection>("GCT");
  produces<l1tp2::CaloTowerCollection>("GCTFullTowers");
  produces<BXVector<l1t::EGamma>>("GCTEGammas");
  produces<l1tp2::DigitizedClusterCorrelatorCollection>("GCTDigitizedClusterToCorrelator");
  produces<l1tp2::DigitizedTowerCorrelatorCollection>("GCTDigitizedTowerToCorrelator");
  produces<l1tp2::DigitizedClusterGTCollection>("GCTDigitizedClusterToGT");
}

Phase2L1CaloEGammaEmulator::~Phase2L1CaloEGammaEmulator() {}

void Phase2L1CaloEGammaEmulator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Detector geometry
  const auto& caloGeometry = iSetup.getData(caloGeometryTag_);
  ebGeometry = caloGeometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  hbGeometry = caloGeometry.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const auto& hbTopology = iSetup.getData(hbTopologyTag_);
  hcTopology_ = &hbTopology;
  HcalTrigTowerGeometry theTrigTowerGeometry(hcTopology_);

  const auto& decoder = iSetup.getData(decoderTag_);

  //***************************************************//
  // Declare RCT output collections
  //***************************************************//

  auto L1EGXtalClusters = std::make_unique<l1tp2::CaloCrystalClusterCollection>();
  auto L1CaloTowers = std::make_unique<l1tp2::CaloTowerCollection>();

  //***************************************************//
  // Get the ECAL hits
  //***************************************************//
  edm::Handle<EcalEBTrigPrimDigiCollection> pcalohits;
  iEvent.getByToken(ecalTPEBToken_, pcalohits);

  std::vector<p2eg::SimpleCaloHit> ecalhits;

  for (const auto& hit : *pcalohits.product()) {
    if (hit.encodedEt() > 0)  // hit.encodedEt() returns an int corresponding to 2x the crystal Et
    {
      // Et is 10 bit, by keeping the ADC saturation Et at 120 GeV it means that you have to divide by 8
      float et = hit.encodedEt() * p2eg::ECAL_LSB;
      if (et < p2eg::cut_500_MeV) {
        continue;  // Reject hits with < 500 MeV ET
      }

      // Get cell coordinates and info
      auto cell = ebGeometry->getGeometry(hit.id());

      p2eg::SimpleCaloHit ehit;
      ehit.setId(hit.id());
      ehit.setPosition(GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z()));
      ehit.setEnergy(et);
      ehit.setEt_uint((ap_uint<10>)hit.encodedEt());  // also save the uint Et
      ehit.setPt();
      ecalhits.push_back(ehit);
    }
  }

  //***************************************************//
  // Get the HCAL hits
  //***************************************************//
  std::vector<p2eg::SimpleCaloHit> hcalhits;
  edm::Handle<edm::SortedCollection<HcalTriggerPrimitiveDigi>> hbhecoll;
  iEvent.getByToken(hcalTPToken_, hbhecoll);

  for (const auto& hit : *hbhecoll.product()) {
    float et = decoder.hcaletValue(hit.id(), hit.t0());
    ap_uint<10> encodedEt = hit.t0().compressedEt();
    // same thing as SOI_compressedEt() in HcalTriggerPrimitiveDigi.h///
    if (et <= 0)
      continue;

    if (!(hcTopology_->validHT(hit.id()))) {
      LogError("Phase2L1CaloEGammaEmulator")
          << " -- Hcal hit DetID not present in HCAL Geom: " << hit.id() << std::endl;
      throw cms::Exception("Phase2L1CaloEGammaEmulator");
      continue;
    }
    const std::vector<HcalDetId>& hcId = theTrigTowerGeometry.detIds(hit.id());
    if (hcId.empty()) {
      LogError("Phase2L1CaloEGammaEmulator") << "Cannot find any HCalDetId corresponding to " << hit.id() << std::endl;
      throw cms::Exception("Phase2L1CaloEGammaEmulator");
      continue;
    }
    if (hcId[0].subdetId() > 1) {
      continue;
    }
    GlobalVector hcal_tp_position = GlobalVector(0., 0., 0.);
    for (const auto& hcId_i : hcId) {
      if (hcId_i.subdetId() > 1) {
        continue;
      }
      // get the first HCAL TP/ cell
      auto cell = hbGeometry->getGeometry(hcId_i);
      if (cell == nullptr) {
        continue;
      }
      GlobalVector tmpVector = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
      hcal_tp_position = tmpVector;

      break;
    }
    p2eg::SimpleCaloHit hhit;
    hhit.setId(hit.id());
    hhit.setIdHcal(hit.id());
    hhit.setPosition(hcal_tp_position);
    hhit.setEnergy(et);
    hhit.setPt();
    hhit.setEt_uint(encodedEt);
    hcalhits.push_back(hhit);
  }

  //***************************************************//
  // Initialize necessary arrays for tower and clusters
  //***************************************************//

  // L1 Outputs definition: Arrays that use firmware convention for indexing
  p2eg::tower_t towerHCALCard
      [p2eg::n_towers_cardEta][p2eg::n_towers_cardPhi]
      [p2eg::n_towers_halfPhi];  // 17x4x36 array (not to be confused with the 12x1 array of ap_uints, towerEtHCAL
  p2eg::tower_t towerECALCard[p2eg::n_towers_cardEta][p2eg::n_towers_cardPhi][p2eg::n_towers_halfPhi];
  // There is one vector of clusters per card (up to 12 clusters before stitching across ECAL regions)
  std::vector<p2eg::Cluster> cluster_list[p2eg::n_towers_halfPhi];
  // After merging/stitching the clusters, we only take the 8 highest pt per card
  std::vector<p2eg::Cluster> cluster_list_merged[p2eg::n_towers_halfPhi];

  //***************************************************//
  // Fill RCT ECAL regions with ECAL hits
  //***************************************************//
  for (int cc = 0; cc < p2eg::n_towers_halfPhi; ++cc) {  // Loop over 36 L1 cards

    p2eg::card rctCard;
    rctCard.setIdx(cc);

    for (const auto& hit : ecalhits) {
      // Check if the hit is in cards 0-35
      if (hit.isInCard(cc)) {
        // Get the crystal eta and phi, relative to the bottom left corner of the card
        // (0 up to 17*5, 0 up to 4*5)
        int local_iEta = hit.crystalLocaliEta(cc);
        int local_iPhi = hit.crystalLocaliPhi(cc);

        // Region number (0-5) depends only on the crystal iEta in the card
        int regionNumber = p2eg::getRegionNumber(local_iEta);

        // Tower eta and phi index inside the card (17x4)
        int inCard_tower_iEta = local_iEta / p2eg::CRYSTALS_IN_TOWER_ETA;
        int inCard_tower_iPhi = local_iPhi / p2eg::CRYSTALS_IN_TOWER_PHI;

        // Tower eta and phi index inside the region (3x4)
        int inRegion_tower_iEta = inCard_tower_iEta % p2eg::TOWER_IN_ETA;
        int inRegion_tower_iPhi = inCard_tower_iPhi % p2eg::TOWER_IN_PHI;

        // Crystal eta and phi index inside the 3x4 region (15x20)
        int inRegion_crystal_iEta = local_iEta % (p2eg::TOWER_IN_ETA * p2eg::CRYSTALS_IN_TOWER_ETA);
        int inRegion_crystal_iPhi = local_iPhi;

        // Crystal eta and phi index inside the tower (5x5)
        int inLink_crystal_iEta = (inRegion_crystal_iEta % p2eg::CRYSTALS_IN_TOWER_ETA);
        int inLink_crystal_iPhi = (inRegion_crystal_iPhi % p2eg::CRYSTALS_IN_TOWER_PHI);

        // Add the crystal energy to the rctCard
        p2eg::region3x4& myRegion = rctCard.getRegion3x4(regionNumber);
        p2eg::linkECAL& myLink = myRegion.getLinkECAL(inRegion_tower_iEta, inRegion_tower_iPhi);
        myLink.addCrystalE(inLink_crystal_iEta, inLink_crystal_iPhi, hit.et_uint());
      }
    }

    //***************************************************//
    // Build RCT towers from HCAL hits
    //***************************************************//
    for (const auto& hit : hcalhits) {
      if (hit.isInCard(cc) && hit.pt() > 0) {
        // Get crystal eta and phi, relative to the bottom left corner of the card
        // (0 up to 17*5, 0 up to 4*5)
        int local_iEta = hit.crystalLocaliEta(cc);
        int local_iPhi = hit.crystalLocaliPhi(cc);

        // Region (0-5) the hit falls into
        int regionNumber = p2eg::getRegionNumber(local_iEta);

        // Tower eta and phi index inside the card (17x4)
        int inCard_tower_iEta = int(local_iEta / p2eg::CRYSTALS_IN_TOWER_ETA);
        int inCard_tower_iPhi = int(local_iPhi / p2eg::CRYSTALS_IN_TOWER_PHI);

        // Tower eta and phi index inside the region (3x4)
        int inRegion_tower_iEta = inCard_tower_iEta % p2eg::TOWER_IN_ETA;
        int inRegion_tower_iPhi = inCard_tower_iPhi % p2eg::TOWER_IN_PHI;

        // Access the right HCAL region and tower and increment the ET
        p2eg::towers3x4& myTowers3x4 = rctCard.getTowers3x4(regionNumber);
        p2eg::towerHCAL& myTower = myTowers3x4.getTowerHCAL(inRegion_tower_iEta, inRegion_tower_iPhi);
        myTower.addEt(hit.et_uint());
      }
    }

    //***************************************************//
    // Make clusters in each ECAL region independently
    //***************************************************//
    for (int idxRegion = 0; idxRegion < p2eg::N_REGIONS_PER_CARD; idxRegion++) {
      // ECAL crystals array
      p2eg::crystal temporary[p2eg::CRYSTAL_IN_ETA][p2eg::CRYSTAL_IN_PHI];
      // HCAL towers in 3x4 region
      ap_uint<12> towerEtHCAL[p2eg::TOWER_IN_ETA * p2eg::TOWER_IN_PHI];

      p2eg::region3x4& myRegion = rctCard.getRegion3x4(idxRegion);
      p2eg::towers3x4& myTowers = rctCard.getTowers3x4(idxRegion);

      // In each 3x4 region, loop through the links (one link = one tower)
      for (int iLinkEta = 0; iLinkEta < p2eg::TOWER_IN_ETA; iLinkEta++) {
        for (int iLinkPhi = 0; iLinkPhi < p2eg::TOWER_IN_PHI; iLinkPhi++) {
          // Get the ECAL link (one link per tower)
          p2eg::linkECAL& myLink = myRegion.getLinkECAL(iLinkEta, iLinkPhi);

          // We have an array of 3x4 links/towers, each link/tower is 5x5 in crystals. We need to convert this to a 15x20 of crystals
          int ref_iEta = (iLinkEta * p2eg::CRYSTALS_IN_TOWER_ETA);
          int ref_iPhi = (iLinkPhi * p2eg::CRYSTALS_IN_TOWER_PHI);

          // In the link, get the crystals (5x5 in each link)
          for (int iEta = 0; iEta < p2eg::CRYSTALS_IN_TOWER_ETA; iEta++) {
            for (int iPhi = 0; iPhi < p2eg::CRYSTALS_IN_TOWER_PHI; iPhi++) {
              // Et as unsigned int
              ap_uint<10> uEnergy = myLink.getCrystalE(iEta, iPhi);

              // Fill the 'temporary' array with a crystal object
              temporary[ref_iEta + iEta][ref_iPhi + iPhi] = p2eg::crystal(uEnergy);
            }
          }  // end of loop over crystals

          // HCAL tower ET
          p2eg::towerHCAL& myTower = myTowers.getTowerHCAL(iLinkEta, iLinkPhi);
          towerEtHCAL[(iLinkEta * p2eg::TOWER_IN_PHI) + iLinkPhi] = myTower.getEt();
        }
      }

      // Iteratively find four clusters and remove them from 'temporary' as we go, and fill cluster_list
      for (int c = 0; c < p2eg::N_CLUSTERS_PER_REGION; c++) {
        p2eg::Cluster newCluster = p2eg::getClusterFromRegion3x4(temporary);  // remove cluster from 'temporary'
        newCluster.setRegionIdx(idxRegion);                                   // add the region number
        if (newCluster.clusterEnergy() > 0) {
          // do not push back 0-energy clusters
          cluster_list[cc].push_back(newCluster);
        }
      }

      // Create towers using remaining ECAL energy, and the HCAL towers were already calculated in towersEtHCAL[12]
      ap_uint<12> towerEtECAL[12];
      p2eg::getECALTowersEt(temporary, towerEtECAL);

      // Fill towerHCALCard and towerECALCard arrays
      for (int i = 0; i < 12; i++) {
        // Get the tower's indices in a (17x4) card
        int iEta = (idxRegion * p2eg::TOWER_IN_ETA) + (i / p2eg::TOWER_IN_PHI);
        int iPhi = (i % p2eg::TOWER_IN_PHI);

        // If the region number is 5 (i.e. only 2x4 in towers, skip the third row) N_REGIONS_PER_CARD = 6. i.e. we do not want to consider
        // i = 8, 9, 10, 11
        if ((idxRegion == (p2eg::N_REGIONS_PER_CARD - 1)) && (i > 7)) {
          continue;
        }
        towerHCALCard[iEta][iPhi][cc] =
            p2eg::tower_t(towerEtHCAL[i], 0);  // p2eg::tower_t initializer takes an ap-uint<12> for the energy
        towerECALCard[iEta][iPhi][cc] = p2eg::tower_t(towerEtECAL[i], 0);
      }
    }

    //-------------------------------------------//
    // Stitching across ECAL regions             //
    //-------------------------------------------//
    const int nRegionBoundariesEta = (p2eg::N_REGIONS_PER_CARD - 1);  // 6 regions -> 5 boundaries to check
    // Upper and lower boundaries respectively, to check for stitching along
    int towerEtaBoundaries[nRegionBoundariesEta][2] = {{15, 14}, {12, 11}, {9, 8}, {6, 5}, {3, 2}};

    for (int iBound = 0; iBound < nRegionBoundariesEta; iBound++) {
      p2eg::stitchClusterOverRegionBoundary(
          cluster_list[cc], towerEtaBoundaries[iBound][0], towerEtaBoundaries[iBound][1], cc);
    }

    //--------------------------------------------------------------------------------//
    // Sort the clusters, take the 8 with highest pT, and return extras to tower
    //--------------------------------------------------------------------------------//
    if (!cluster_list[cc].empty()) {
      std::sort(cluster_list[cc].begin(), cluster_list[cc].end(), p2eg::compareClusterET);

      // If there are more than eight clusters, return the unused energy to the towers
      for (unsigned int kk = p2eg::n_clusters_4link; kk < cluster_list[cc].size(); ++kk) {
        p2eg::Cluster cExtra = cluster_list[cc][kk];
        if (cExtra.clusterEnergy() > 0) {
          // Increment tower ET
          // Get tower (eta, phi) (up to (17, 4)) in the RCT card
          int whichTowerEtaInCard = ((cExtra.region() * p2eg::TOWER_IN_ETA) + cExtra.towerEta());
          int whichTowerPhiInCard = cExtra.towerPhi();
          ap_uint<12> oldTowerEt = towerECALCard[whichTowerEtaInCard][whichTowerPhiInCard][cc].et();
          ap_uint<12> newTowerEt = (oldTowerEt + cExtra.clusterEnergy());
          ap_uint<4> hoe = towerECALCard[whichTowerEtaInCard][whichTowerPhiInCard][cc].hoe();
          towerECALCard[whichTowerEtaInCard][whichTowerPhiInCard][cc] = p2eg::tower_t(newTowerEt, hoe);
        }
      }

      // Save up to eight clusters: loop over cluster_list
      for (unsigned int kk = 0; kk < cluster_list[cc].size(); ++kk) {
        if (kk >= p2eg::n_clusters_4link)
          continue;
        if (cluster_list[cc][kk].clusterEnergy() > 0) {
          cluster_list_merged[cc].push_back(cluster_list[cc][kk]);
        }
      }
    }

    //-------------------------------------------//
    // Calibrate clusters
    //-------------------------------------------//
    for (auto& c : cluster_list_merged[cc]) {
      float realEta = c.realEta(cc);
      c.calib = calib_(c.getPt(), std::abs(realEta));
      c.applyCalibration(c.calib);
    }

    //-------------------------------------------//
    // Cluster shower shape flags
    //-------------------------------------------//
    for (auto& c : cluster_list_merged[cc]) {
      c.is_ss = p2eg::passes_ss(c.getPt(), (c.getEt2x5() / c.getEt5x5()));
      c.is_looseTkss = p2eg::passes_looseTkss(c.getPt(), (c.getEt2x5() / c.getEt5x5()));
    }

    //-------------------------------------------//
    // Calibrate towers
    //-------------------------------------------//
    for (int ii = 0; ii < p2eg::n_towers_cardEta; ++ii) {    // 17 towers per card in eta
      for (int jj = 0; jj < p2eg::n_towers_cardPhi; ++jj) {  // 4 towers per card in phi
        float tRealEta = p2eg::getTowerEta_fromAbsID(
            p2eg::getAbsID_iEta_fromFirmwareCardTowerLink(cc, ii, jj));  // real eta of center of tower
        double tCalib = calib_(0, tRealEta);                             // calibration factor
        towerECALCard[ii][jj][cc].applyCalibration(tCalib);
      }
    }

    //-------------------------------------------//
    // Calculate tower HoE
    //-------------------------------------------//
    for (int ii = 0; ii < p2eg::n_towers_cardEta; ++ii) {    // 17 towers per card in eta
      for (int jj = 0; jj < p2eg::n_towers_cardPhi; ++jj) {  // 4 towers per card in phi
        ap_uint<12> ecalEt = towerECALCard[ii][jj][cc].et();
        ap_uint<12> hcalEt = towerHCALCard[ii][jj][cc].et();
        towerECALCard[ii][jj][cc].getHoverE(ecalEt, hcalEt);
      }
    }

    //-----------------------------------------------------------//
    // Produce output RCT collections for event display and analyzer
    //-----------------------------------------------------------//
    for (auto& c : cluster_list_merged[cc]) {
      reco::Candidate::PolarLorentzVector p4calibrated(c.getPt(), c.realEta(cc), c.realPhi(cc), 0.);

      // Constructor definition at: https://github.com/cms-l1t-offline/cmssw/blob/l1t-phase2-v3.3.11/DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h#L34
      l1tp2::CaloCrystalCluster cluster(p4calibrated,
                                        c.getPt(),           // use float
                                        0,                   // float h over e
                                        0,                   // float iso
                                        0,                   // DetId seedCrystal
                                        0,                   // puCorrPt
                                        c.getBrems(),        // 0, 1, or 2 (as computed in firmware)
                                        0,                   // et2x2 (not calculated)
                                        c.getEt2x5(),        // et2x5 (as computed in firmware, save float)
                                        0,                   // et3x5 (not calculated)
                                        c.getEt5x5(),        // et5x5 (as computed in firmware, save float)
                                        c.getIsSS(),         // standalone WP
                                        c.getIsSS(),         // electronWP98
                                        false,               // photonWP80
                                        c.getIsSS(),         // electronWP90
                                        c.getIsLooseTkss(),  // looseL1TkMatchWP
                                        c.getIsSS()          // stage2effMatch
      );

      std::map<std::string, float> params;
      params["standaloneWP_showerShape"] = c.getIsSS();
      params["trkMatchWP_showerShape"] = c.getIsLooseTkss();
      cluster.setExperimentalParams(params);

      L1EGXtalClusters->push_back(cluster);
    }
    // Output tower collections
    for (int ii = 0; ii < p2eg::n_towers_cardEta; ++ii) {    // 17 towers per card in eta
      for (int jj = 0; jj < p2eg::n_towers_cardPhi; ++jj) {  // 4 towers per card in phi

        l1tp2::CaloTower l1CaloTower;
        // Divide by 8.0 to get ET as float (GeV)
        l1CaloTower.setEcalTowerEt(towerECALCard[ii][jj][cc].et() * p2eg::ECAL_LSB);
        // HCAL TPGs encoded ET: multiply by the LSB (0.5) to convert to GeV
        l1CaloTower.setHcalTowerEt(towerHCALCard[ii][jj][cc].et() * p2eg::HCAL_LSB);
        int absToweriEta = p2eg::getAbsID_iEta_fromFirmwareCardTowerLink(cc, ii, jj);
        int absToweriPhi = p2eg::getAbsID_iPhi_fromFirmwareCardTowerLink(cc, ii, jj);
        l1CaloTower.setTowerIEta(absToweriEta);
        l1CaloTower.setTowerIPhi(absToweriPhi);
        l1CaloTower.setTowerEta(p2eg::getTowerEta_fromAbsID(absToweriEta));
        l1CaloTower.setTowerPhi(p2eg::getTowerPhi_fromAbsID(absToweriPhi));

        L1CaloTowers->push_back(l1CaloTower);
      }
    }
  }  // end of loop over cards

  iEvent.put(std::move(L1EGXtalClusters), "RCT");
  iEvent.put(std::move(L1CaloTowers), "RCT");

  //*******************************************************************
  // Do GCT geometry for inputs
  //*******************************************************************

  // Loop over GCT cards (three of them)
  p2eg::GCTcard_t gctCards[p2eg::N_GCTCARDS];
  p2eg::GCTtoCorr_t gctToCorr[p2eg::N_GCTCARDS];

  // Initialize the cards (requires towerECALCard, towerHCALCard arrays, and cluster_list_merged)
  for (unsigned int gcc = 0; gcc < p2eg::N_GCTCARDS; gcc++) {
    // Each GCT card encompasses 16 RCT cards, listed in GCTcardtoRCTcardnumber[3][16]. i goes from 0 to <16
    for (int i = 0; i < (p2eg::N_RCTCARDS_PHI * 2); i++) {
      unsigned int rcc = p2eg::GCTcardtoRCTcardnumber[gcc][i];

      // Positive eta? Fist row is in positive eta
      bool isPositiveEta = (i < p2eg::N_RCTCARDS_PHI);

      // Sum tower ECAL and HCAL energies: 17 towers per link
      for (int iTower = 0; iTower < p2eg::N_GCTTOWERS_FIBER; iTower++) {
        // 4 links per card
        for (int iLink = 0; iLink < p2eg::n_links_card; iLink++) {
          p2eg::tower_t t0_ecal = towerECALCard[iTower][iLink][rcc];
          p2eg::tower_t t0_hcal = towerHCALCard[iTower][iLink][rcc];
          p2eg::RCTtower_t t;
          t.et = t0_ecal.et() + p2eg::convertHcalETtoEcalET(t0_hcal.et());
          t.hoe = t0_ecal.hoe();
          // Not needed for GCT firmware but will be written into GCT CMSSW outputs : 12 bits each
          t.ecalEt = t0_ecal.et();
          t.hcalEt = t0_hcal.et();

          if (isPositiveEta) {
            gctCards[gcc].RCTcardEtaPos[i % p2eg::N_RCTCARDS_PHI].RCTtoGCTfiber[iLink].RCTtowers[iTower] = t;
          } else {
            gctCards[gcc].RCTcardEtaNeg[i % p2eg::N_RCTCARDS_PHI].RCTtoGCTfiber[iLink].RCTtowers[iTower] = t;
          }
        }
      }

      // Distribute at most 8 RCT clusters across four links: convert to GCT coordinates
      for (size_t iCluster = 0; (iCluster < cluster_list_merged[rcc].size()) &&
                                (iCluster < (p2eg::N_RCTGCT_FIBERS * p2eg::N_RCTCLUSTERS_FIBER));
           iCluster++) {
        p2eg::Cluster c0 = cluster_list_merged[rcc][iCluster];
        p2eg::RCTcluster_t c;
        c.et = c0.clusterEnergy();

        // tower Eta: c0.towerEta() refers to the tower iEta INSIDE the region, so we need to convert to tower iEta inside the card
        c.towEta = (c0.region() * p2eg::TOWER_IN_ETA) + c0.towerEta();
        c.towPhi = c0.towerPhi();
        c.crEta = c0.clusterEta();
        c.crPhi = c0.clusterPhi();
        c.et5x5 = c0.uint_et5x5();
        c.et2x5 = c0.uint_et2x5();
        c.is_ss = c0.getIsSS();
        c.is_looseTkss = c0.getIsLooseTkss();
        c.is_iso = c0.getIsIso();
        c.is_looseTkiso = c0.getIsLooseTkIso();
        c.brems = c0.getBrems();
        c.nGCTCard = gcc;  // store gct card index as well
        unsigned int iIdxInGCT = i % p2eg::N_RCTCARDS_PHI;
        unsigned int iLinkC = iCluster % p2eg::N_RCTGCT_FIBERS;
        unsigned int iPosC = iCluster / p2eg::N_RCTGCT_FIBERS;

        if (isPositiveEta) {
          gctCards[gcc].RCTcardEtaPos[iIdxInGCT].RCTtoGCTfiber[iLinkC].RCTclusters[iPosC] = c;
        } else {
          gctCards[gcc].RCTcardEtaNeg[iIdxInGCT].RCTtoGCTfiber[iLinkC].RCTclusters[iPosC] = c;
        }
      }
      // If there were fewer than eight clusters, make sure the remaining fiber clusters are zero'd out.
      for (size_t iZeroCluster = cluster_list_merged[rcc].size();
           iZeroCluster < (p2eg::N_RCTGCT_FIBERS * p2eg::N_RCTCLUSTERS_FIBER);
           iZeroCluster++) {
        unsigned int iIdxInGCT = i % p2eg::N_RCTCARDS_PHI;
        unsigned int iLinkC = iZeroCluster % p2eg::N_RCTGCT_FIBERS;
        unsigned int iPosC = iZeroCluster / p2eg::N_RCTGCT_FIBERS;

        p2eg::RCTcluster_t cZero;
        cZero.et = 0;
        cZero.towEta = 0;
        cZero.towPhi = 0;
        cZero.crEta = 0;
        cZero.crPhi = 0;
        cZero.et5x5 = 0;
        cZero.et2x5 = 0;
        cZero.is_ss = false;
        cZero.is_looseTkss = false;
        cZero.is_iso = false;
        cZero.is_looseTkiso = false;
        cZero.nGCTCard = gcc;  // store gct card index as well
        if (isPositiveEta) {
          gctCards[gcc].RCTcardEtaPos[iIdxInGCT].RCTtoGCTfiber[iLinkC].RCTclusters[iPosC] = cZero;
        } else {
          gctCards[gcc].RCTcardEtaNeg[iIdxInGCT].RCTtoGCTfiber[iLinkC].RCTclusters[iPosC] = cZero;
        }
      }
    }
  }  // end of loop over initializing GCT cards

  //----------------------------------------------------
  // Output collections for the GCT emulator
  //----------------------------------------------------
  auto L1GCTClusters = std::make_unique<l1tp2::CaloCrystalClusterCollection>();
  auto L1GCTTowers = std::make_unique<l1tp2::CaloTowerCollection>();
  auto L1GCTFullTowers = std::make_unique<l1tp2::CaloTowerCollection>();
  auto L1GCTEGammas = std::make_unique<l1t::EGammaBxCollection>();
  auto L1DigitizedClusterCorrelator = std::make_unique<l1tp2::DigitizedClusterCorrelatorCollection>();
  auto L1DigitizedTowerCorrelator = std::make_unique<l1tp2::DigitizedTowerCorrelatorCollection>();
  auto L1DigitizedClusterGT = std::make_unique<l1tp2::DigitizedClusterGTCollection>();

  //----------------------------------------------------
  // Apply the GCT firmware code to each GCT card
  //----------------------------------------------------

  for (unsigned int gcc = 0; gcc < p2eg::N_GCTCARDS; gcc++) {
    p2eg::algo_top(gctCards[gcc],
                   gctToCorr[gcc],
                   gcc,
                   L1GCTClusters,
                   L1GCTTowers,
                   L1GCTFullTowers,
                   L1GCTEGammas,
                   L1DigitizedClusterCorrelator,
                   L1DigitizedTowerCorrelator,
                   L1DigitizedClusterGT,
                   calib_);
  }

  iEvent.put(std::move(L1GCTClusters), "GCT");
  iEvent.put(std::move(L1GCTTowers), "GCT");
  iEvent.put(std::move(L1GCTFullTowers), "GCTFullTowers");
  iEvent.put(std::move(L1GCTEGammas), "GCTEGammas");
  iEvent.put(std::move(L1DigitizedClusterCorrelator), "GCTDigitizedClusterToCorrelator");
  iEvent.put(std::move(L1DigitizedTowerCorrelator), "GCTDigitizedTowerToCorrelator");
  iEvent.put(std::move(L1DigitizedClusterGT), "GCTDigitizedClusterToGT");
}

//////////////////////////////////////////////////////////////////////////

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1CaloEGammaEmulator);
