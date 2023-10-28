// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1EGammaCrystalsEmulatorProducer
//
/**\class L1EGammaCrystalsEmulatorProducer L1EGammaCrystalsEmulatorProducer.cc L1Trigger/L1CaloTrigger/plugins/L1EGammaCrystalsEmulatorProducer.cc
Description: Produces crystal clusters using crystal-level information and hardware constraints
Implementation:
[Notes on implementation]
*/
//
// Original Author: Cecile Caillol
// Created: Tue Aug 10 2018
//
// $Id$
//
//

// system include files
#include <memory>
#include <array>
#include <iostream>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
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
#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "L1Trigger/L1CaloTrigger/interface/ParametricCalibration.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

static constexpr bool do_brem = true;

static constexpr int n_eta_bins = 2;
static constexpr int n_borders_phi = 18;
static constexpr int n_borders_eta = 18;
static constexpr int n_clusters_max = 5;
static constexpr int n_clusters_link = 3;
static constexpr int n_clusters_4link = 4 * 3;
static constexpr int n_crystals_towerEta = 5;
static constexpr int n_crystals_towerPhi = 5;
static constexpr int n_crystals_3towers = 3 * 5;
static constexpr int n_crystals_2towers = 2 * 5;
static constexpr int n_towers_per_link = 17;
static constexpr int n_clusters_per_link = 2;
static constexpr int n_clusters_per_L1card = 8;
static constexpr int n_towers_Eta = 34;
static constexpr int n_towers_Phi = 72;
static constexpr int n_towers_halfPhi = 36;
static constexpr int n_links_card = 4;
static constexpr int n_links_GCTcard = 48;
static constexpr int n_GCTcards = 3;
static constexpr float ECAL_eta_range = 1.4841;
static constexpr float half_crystal_size = 0.00873;
static constexpr float slideIsoPtThreshold = 80;
static constexpr float plateau_ss = 130.0;
static constexpr float a0_80 = 0.85, a1_80 = 0.0080, a0 = 0.21;                        // passes_iso
static constexpr float b0 = 0.38, b1 = 1.9, b2 = 0.05;                                 //passes_looseTkiso
static constexpr float c0_ss = 0.94, c1_ss = 0.052, c2_ss = 0.044;                     //passes_ss
static constexpr float d0 = 0.96, d1 = 0.0003;                                         //passes_photon
static constexpr float e0_looseTkss = 0.944, e1_looseTkss = 0.65, e2_looseTkss = 0.4;  //passes_looseTkss
static constexpr float cut_500_MeV = 0.5;

// absolue IDs range from 0-33
// 0-16 are iEta -17 to -1
// 17 to 33 are iEta 1 to 17
static constexpr int toweriEta_fromAbsoluteID_shift = 16;

// absolue IDs range from 0-71.
// To align with detector tower IDs (1 - n_towers_Phi)
// shift all indices by 37 and loop over after 72
static constexpr int toweriPhi_fromAbsoluteID_shift_lowerHalf = 37;
static constexpr int toweriPhi_fromAbsoluteID_shift_upperHalf = 35;

float getEta_fromL2LinkCardTowerCrystal(int link, int card, int tower, int crystal) {
  int etaID = n_crystals_towerEta * (n_towers_per_link * ((link / n_links_card) % 2) + (tower % n_towers_per_link)) +
              crystal % n_crystals_towerEta;
  float size_cell = 2 * ECAL_eta_range / (n_crystals_towerEta * n_towers_Eta);
  return etaID * size_cell - ECAL_eta_range + half_crystal_size;
}

float getPhi_fromL2LinkCardTowerCrystal(int link, int card, int tower, int crystal) {
  int phiID = n_crystals_towerPhi * ((card * 24) + (n_links_card * (link / 8)) + (tower / n_towers_per_link)) +
              crystal / n_crystals_towerPhi;
  float size_cell = 2 * M_PI / (n_crystals_towerPhi * n_towers_Phi);
  return phiID * size_cell - M_PI + half_crystal_size;
}

int getCrystal_etaID(float eta) {
  float size_cell = 2 * ECAL_eta_range / (n_crystals_towerEta * n_towers_Eta);
  int etaID = int((eta + ECAL_eta_range) / size_cell);
  return etaID;
}

int convert_L2toL1_link(int link) { return link % n_links_card; }

int convert_L2toL1_tower(int tower) { return tower; }

int convert_L2toL1_card(int card, int link) { return card * n_clusters_4link + link / n_links_card; }

int getCrystal_phiID(float phi) {
  float size_cell = 2 * M_PI / (n_crystals_towerPhi * n_towers_Phi);
  int phiID = int((phi + M_PI) / size_cell);
  return phiID;
}

int getTower_absoluteEtaID(float eta) {
  float size_cell = 2 * ECAL_eta_range / n_towers_Eta;
  int etaID = int((eta + ECAL_eta_range) / size_cell);
  return etaID;
}

int getTower_absolutePhiID(float phi) {
  float size_cell = 2 * M_PI / n_towers_Phi;
  int phiID = int((phi + M_PI) / size_cell);
  return phiID;
}

int getToweriEta_fromAbsoluteID(int id) {
  if (id < n_towers_per_link)
    return id - n_towers_per_link;
  else
    return id - toweriEta_fromAbsoluteID_shift;
}

int getToweriPhi_fromAbsoluteID(int id) {
  if (id < n_towers_Phi / 2)
    return id + toweriPhi_fromAbsoluteID_shift_lowerHalf;
  else
    return id - toweriPhi_fromAbsoluteID_shift_upperHalf;
}

float getTowerEta_fromAbsoluteID(int id) {
  float size_cell = 2 * ECAL_eta_range / n_towers_Eta;
  float eta = (id * size_cell) - ECAL_eta_range + 0.5 * size_cell;
  return eta;
}

float getTowerPhi_fromAbsoluteID(int id) {
  float size_cell = 2 * M_PI / n_towers_Phi;
  float phi = (id * size_cell) - M_PI + 0.5 * size_cell;
  return phi;
}

int getCrystalIDInTower(int etaID, int phiID) {
  return int(n_crystals_towerPhi * (phiID % n_crystals_towerPhi) + (etaID % n_crystals_towerEta));
}

int get_towerEta_fromCardTowerInCard(int card, int towerincard) {
  return n_towers_per_link * (card % 2) + towerincard % n_towers_per_link;
}

int get_towerPhi_fromCardTowerInCard(int card, int towerincard) {
  return 4 * (card / 2) + towerincard / n_towers_per_link;
}

int get_towerEta_fromCardLinkTower(int card, int link, int tower) { return n_towers_per_link * (card % 2) + tower; }

int get_towerPhi_fromCardLinkTower(int card, int link, int tower) { return 4 * (card / 2) + link; }

int getTowerID(int etaID, int phiID) {
  return int(n_towers_per_link * ((phiID / n_crystals_towerPhi) % 4) +
             (etaID / n_crystals_towerEta) % n_towers_per_link);
}

int getTower_phiID(int cluster_phiID) {  // Tower ID in card given crystal ID in total detector
  return int((cluster_phiID / n_crystals_towerPhi) % 4);
}

int getTower_etaID(int cluster_etaID) {  // Tower ID in card given crystal ID in total detector
  return int((cluster_etaID / n_crystals_towerEta) % n_towers_per_link);
}

int getEtaMax_card(int card) {
  int etamax = 0;
  if (card % 2 == 0)
    etamax = n_towers_per_link * n_crystals_towerEta - 1;  // First eta half. 5 crystals in eta in 1 tower.
  else
    etamax = n_towers_Eta * n_crystals_towerEta - 1;
  return etamax;
}

int getEtaMin_card(int card) {
  int etamin = 0;
  if (card % 2 == 0)
    etamin = 0 * n_crystals_towerEta;  // First eta half. 5 crystals in eta in 1 tower.
  else
    etamin = n_towers_per_link * n_crystals_towerEta;
  return etamin;
}

int getPhiMax_card(int card) {
  int phimax = ((card / 2) + 1) * 4 * n_crystals_towerPhi - 1;
  return phimax;
}

int getPhiMin_card(int card) {
  int phimin = (card / 2) * 4 * n_crystals_towerPhi;
  return phimin;
}

/* 
 * Replace in-line region boundary arithmetic with function that accounts for region 0 in negative eta cards
 * In the indexing convention of the old emulator,  region 0 is the region overlapping with the endcap, and is
 * only two towers wide in eta.
 */
int getEtaMin_region(int card, int nregion) {
  // Special handling for negative-eta cards
  if (card % 2 == 0) {
    if (nregion == 0) {
      return getEtaMin_card(card);
    } else {
      return getEtaMin_card(card) + n_crystals_2towers + n_crystals_3towers * (nregion - 1);
    }
  }
  // Positive-eta cards: same as original in-line arithmetic
  else {
    return getEtaMin_card(card) + n_crystals_3towers * nregion;
  }
}

/* 
 * Replace in-line region boundary arithmetic that accounts for region 0 in negative eta cards.
 * Same as above but for max eta of the region. 
 */
int getEtaMax_region(int card, int nregion) {
  // Special handling for negative-eta cards
  if (card % 2 == 0) {
    if (nregion == 0) {
      return getEtaMin_card(card) + n_crystals_2towers;
    } else {
      return getEtaMin_card(card) + n_crystals_2towers + (n_crystals_3towers * nregion);
    }
  }
  // Positive-eta cards: same as original in-line arithmetic
  else {
    return getEtaMin_card(card) + n_crystals_3towers * (nregion + 1);
  }
}

class L1EGCrystalClusterEmulatorProducer : public edm::stream::EDProducer<> {
public:
  explicit L1EGCrystalClusterEmulatorProducer(const edm::ParameterSet&);
  ~L1EGCrystalClusterEmulatorProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  bool passes_ss(float pt, float ss);
  bool passes_photon(float pt, float pss);
  bool passes_iso(float pt, float iso);
  bool passes_looseTkss(float pt, float ss);
  bool passes_looseTkiso(float pt, float iso);
  float get_calibrate(float uncorr);

  edm::EDGetTokenT<EcalEBTrigPrimDigiCollection> ecalTPEBToken_;
  edm::EDGetTokenT<edm::SortedCollection<HcalTriggerPrimitiveDigi> > hcalTPToken_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> decoderTag_;

  l1tp2::ParametricCalibration calib_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryTag_;
  const CaloSubdetectorGeometry* ebGeometry;
  const CaloSubdetectorGeometry* hbGeometry;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hbTopologyTag_;
  const HcalTopology* hcTopology_;

  struct mycluster {
    float c2x2_;
    float c2x5_;
    float c5x5_;
    int cshowershape_;
    int cshowershapeloosetk_;
    float cvalueshowershape_;
    int cphotonshowershape_;
    float cpt;   // ECAL pt
    int cbrem_;  // if brem corrections were applied
    float cWeightedEta_;
    float cWeightedPhi_;
    float ciso_;      // pt of cluster divided by 7x7 ECAL towers
    float crawIso_;   // raw isolation sum
    float chovere_;   // 5x5 HCAL towers divided by the ECAL cluster pt
    float craweta_;   // coordinates between -1.44 and 1.44
    float crawphi_;   // coordinates between -pi and pi
    float chcal_;     // 5x5 HCAL towers
    float ceta_;      // eta ID in the whole detector (between 0 and 5*34-1)
    float cphi_;      // phi ID in the whole detector (between 0 and 5*72-1)
    int ccrystalid_;  // crystal ID inside tower (between 0 and 24)
    int cinsidecrystalid_;
    int ctowerid_;  // tower ID inside card (between 0 and 4*n_towers_per_link-1)
  };

  class SimpleCaloHit {
  private:
    float pt_ = 0;
    float energy_ = 0.;
    bool isEndcapHit_ = false;  // If using endcap, we won't be using integer crystal indices
    bool stale_ = false;        // Hits become stale once used in clustering algorithm to prevent overlap in clusters
    bool used_ = false;
    GlobalVector position_;  // As opposed to GlobalPoint, so we can add them (for weighted average)
    HcalDetId id_hcal_;
    EBDetId id_;

  public:
    // tool functions
    inline void setPt() { pt_ = (position_.mag2() > 0) ? energy_ * sin(position_.theta()) : 0; };
    inline void setEnergy(float et) { energy_ = et / sin(position_.theta()); };
    inline void setIsEndcapHit(bool isEC) { isEndcapHit_ = isEC; };
    inline void setUsed(bool isUsed) { used_ = isUsed; };
    inline void setPosition(const GlobalVector& pos) { position_ = pos; };
    inline void setIdHcal(const HcalDetId& idhcal) { id_hcal_ = idhcal; };
    inline void setId(const EBDetId& id) { id_ = id; };

    inline float pt() const { return pt_; };
    inline float energy() const { return energy_; };
    inline bool isEndcapHit() const { return isEndcapHit_; };
    inline bool used() const { return used_; };
    inline const GlobalVector& position() const { return position_; };
    inline const EBDetId& id() const { return id_; };

    /*
     * Check if it falls within the boundary of a card.
     */
    bool isInCard(int cc) const {
      return (getCrystal_phiID(position().phi()) <= getPhiMax_card(cc) &&
              getCrystal_phiID(position().phi()) >= getPhiMin_card(cc) &&
              getCrystal_etaID(position().eta()) <= getEtaMax_card(cc) &&
              getCrystal_etaID(position().eta()) >= getEtaMin_card(cc));
    };

    /* 
     * Check if it falls within the boundary card AND a region in the card.
     */
    bool isInCardAndRegion(int cc, int nregion) const {
      bool isInRegionEta = (getCrystal_etaID(position().eta()) < getEtaMax_region(cc, nregion) &&
                            getCrystal_etaID(position().eta()) >= getEtaMin_region(cc, nregion));
      return (isInCard(cc) && isInRegionEta);
    }

    // Comparison functions with other SimpleCaloHit instances
    inline float deta(SimpleCaloHit& other) const { return position_.eta() - other.position().eta(); };
    int dieta(SimpleCaloHit& other) const {
      if (isEndcapHit_ || other.isEndcapHit())
        return 9999;  // We shouldn't compare integer indices in endcap, the map is not linear
      if (id_.ieta() * other.id().ieta() > 0)
        return id_.ieta() - other.id().ieta();
      return id_.ieta() - other.id().ieta() - 1;
    };
    inline float dphi(SimpleCaloHit& other) const {
      return reco::deltaPhi(static_cast<float>(position_.phi()), static_cast<float>(other.position().phi()));
    };
    int diphi(SimpleCaloHit& other) const {
      if (isEndcapHit_ || other.isEndcapHit())
        return 9999;  // We shouldn't compare integer indices in endcap, the map is not linear
      // Logic from EBDetId::distancePhi() without the abs()
      static constexpr int PI = 180;
      int result = id().iphi() - other.id().iphi();
      while (result > PI)
        result -= 2 * PI;
      while (result <= -PI)
        result += 2 * PI;
      return result;
    };
    int dieta_byCrystalID(SimpleCaloHit& other) const {
      return (getCrystal_etaID(position_.eta()) - getCrystal_etaID(other.position().eta()));
    };
    int diphi_byCrystalID(SimpleCaloHit& other) const {
      return (getCrystal_phiID(position_.phi()) - getCrystal_phiID(other.position().phi()));
    };
    int id_iEta() { return id_.ieta(); }
    int id_iPhi() { return id_.iphi(); }
    float distanceTo(SimpleCaloHit& other) const {
      // Treat position as a point, measure 3D distance
      // This is used for endcap hits, where we don't have a rectangular mapping
      return (position() - other.position()).mag();
    };
    bool operator==(SimpleCaloHit& other) const {
      return (id_ == other.id() && position() == other.position() && energy_ == other.energy() &&
              isEndcapHit_ == other.isEndcapHit());
    };
  };
};

L1EGCrystalClusterEmulatorProducer::L1EGCrystalClusterEmulatorProducer(const edm::ParameterSet& iConfig)
    : ecalTPEBToken_(consumes<EcalEBTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalTPEB"))),
      hcalTPToken_(
          consumes<edm::SortedCollection<HcalTriggerPrimitiveDigi> >(iConfig.getParameter<edm::InputTag>("hcalTP"))),
      decoderTag_(esConsumes<CaloTPGTranscoder, CaloTPGRecord>(edm::ESInputTag("", ""))),
      calib_(iConfig.getParameter<edm::ParameterSet>("calib")),
      caloGeometryTag_(esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag("", ""))),
      hbTopologyTag_(esConsumes<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag("", ""))) {
  produces<l1tp2::CaloCrystalClusterCollection>();
  produces<BXVector<l1t::EGamma> >();
  produces<l1tp2::CaloTowerCollection>("L1CaloTowerCollection");
}

L1EGCrystalClusterEmulatorProducer::~L1EGCrystalClusterEmulatorProducer() {}

void L1EGCrystalClusterEmulatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<EcalEBTrigPrimDigiCollection> pcalohits;
  iEvent.getByToken(ecalTPEBToken_, pcalohits);

  const auto& caloGeometry = iSetup.getData(caloGeometryTag_);
  ebGeometry = caloGeometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  hbGeometry = caloGeometry.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  const auto& hbTopology = iSetup.getData(hbTopologyTag_);
  hcTopology_ = &hbTopology;
  HcalTrigTowerGeometry theTrigTowerGeometry(hcTopology_);

  const auto& decoder = iSetup.getData(decoderTag_);

  //****************************************************************
  //******************* Get all the hits ***************************
  //****************************************************************

  // Get all the ECAL hits
  iEvent.getByToken(ecalTPEBToken_, pcalohits);
  std::vector<SimpleCaloHit> ecalhits;

  for (const auto& hit : *pcalohits.product()) {
    if (hit.encodedEt() > 0)  // hit.encodedEt() returns an int corresponding to 2x the crystal Et
    {
      // Et is 10 bit, by keeping the ADC saturation Et at 120 GeV it means that you have to divide by 8
      float et = hit.encodedEt() / 8.;
      if (et < cut_500_MeV)
        continue;  // keep the 500 MeV ET Cut

      auto cell = ebGeometry->getGeometry(hit.id());

      SimpleCaloHit ehit;
      ehit.setId(hit.id());
      ehit.setPosition(GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z()));
      ehit.setEnergy(et);
      ehit.setPt();
      ecalhits.push_back(ehit);
    }
  }

  // Get all the HCAL hits
  std::vector<SimpleCaloHit> hcalhits;
  edm::Handle<edm::SortedCollection<HcalTriggerPrimitiveDigi> > hbhecoll;
  iEvent.getByToken(hcalTPToken_, hbhecoll);
  for (const auto& hit : *hbhecoll.product()) {
    float et = decoder.hcaletValue(hit.id(), hit.t0());
    if (et <= 0)
      continue;
    if (!(hcTopology_->validHT(hit.id()))) {
      LogError("L1EGCrystalClusterEmulatorProducer")
          << " -- Hcal hit DetID not present in HCAL Geom: " << hit.id() << std::endl;
      throw cms::Exception("L1EGCrystalClusterEmulatorProducer");
      continue;
    }
    const std::vector<HcalDetId>& hcId = theTrigTowerGeometry.detIds(hit.id());
    if (hcId.empty()) {
      LogError("L1EGCrystalClusterEmulatorProducer")
          << "Cannot find any HCalDetId corresponding to " << hit.id() << std::endl;
      throw cms::Exception("L1EGCrystalClusterEmulatorProducer");
      continue;
    }
    if (hcId[0].subdetId() > 1)
      continue;
    GlobalVector hcal_tp_position = GlobalVector(0., 0., 0.);
    for (const auto& hcId_i : hcId) {
      if (hcId_i.subdetId() > 1)
        continue;
      auto cell = hbGeometry->getGeometry(hcId_i);
      if (cell == nullptr)
        continue;
      GlobalVector tmpVector = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
      hcal_tp_position = tmpVector;
      break;
    }
    SimpleCaloHit hhit;
    hhit.setId(hit.id());
    hhit.setIdHcal(hit.id());
    hhit.setPosition(hcal_tp_position);
    hhit.setEnergy(et);
    hhit.setPt();
    hcalhits.push_back(hhit);
  }

  //*******************************************************************
  //********************** Do layer 1 *********************************
  //*******************************************************************

  // Definition of L1 outputs
  // 36 L1 cards send each 4 links with 17 towers
  float ECAL_tower_L1Card[n_links_card][n_towers_per_link][n_towers_halfPhi];
  float HCAL_tower_L1Card[n_links_card][n_towers_per_link][n_towers_halfPhi];
  int iEta_tower_L1Card[n_links_card][n_towers_per_link][n_towers_halfPhi];
  int iPhi_tower_L1Card[n_links_card][n_towers_per_link][n_towers_halfPhi];
  // 36 L1 cards send each 4 links with 3 clusters
  float energy_cluster_L1Card[n_links_card][n_clusters_link][n_towers_halfPhi];
  // 36 L1 cards send each 4 links with 3 clusters
  int brem_cluster_L1Card[n_links_card][n_clusters_link][n_towers_halfPhi];
  int towerID_cluster_L1Card[n_links_card][n_clusters_link][n_towers_halfPhi];
  int crystalID_cluster_L1Card[n_links_card][n_clusters_link][n_towers_halfPhi];
  int showerShape_cluster_L1Card[n_links_card][n_clusters_link][n_towers_halfPhi];
  int showerShapeLooseTk_cluster_L1Card[n_links_card][n_clusters_link][n_towers_halfPhi];
  int photonShowerShape_cluster_L1Card[n_links_card][n_clusters_link][n_towers_halfPhi];

  for (int ii = 0; ii < n_links_card; ++ii) {
    for (int jj = 0; jj < n_towers_per_link; ++jj) {
      for (int ll = 0; ll < n_towers_halfPhi; ++ll) {
        ECAL_tower_L1Card[ii][jj][ll] = 0;
        HCAL_tower_L1Card[ii][jj][ll] = 0;
        iPhi_tower_L1Card[ii][jj][ll] = -999;
        iEta_tower_L1Card[ii][jj][ll] = -999;
      }
    }
  }
  for (int ii = 0; ii < n_links_card; ++ii) {
    for (int jj = 0; jj < n_clusters_link; ++jj) {
      for (int ll = 0; ll < n_towers_halfPhi; ++ll) {
        energy_cluster_L1Card[ii][jj][ll] = 0;
        brem_cluster_L1Card[ii][jj][ll] = 0;
        towerID_cluster_L1Card[ii][jj][ll] = 0;
        crystalID_cluster_L1Card[ii][jj][ll] = 0;
      }
    }
  }

  // There is one list of clusters per card. We take the 12 highest pt per card
  vector<mycluster> cluster_list[n_towers_halfPhi];
  // After merging the clusters in different regions of a single L1 card
  vector<mycluster> cluster_list_merged[n_towers_halfPhi];

  for (int cc = 0; cc < n_towers_halfPhi; ++cc) {  // Loop over 36 L1 cards
    // Loop over 3x4 etaxphi regions to search for max 5 clusters
    for (int nregion = 0; nregion <= n_clusters_max; ++nregion) {
      int nclusters = 0;
      bool build_cluster = true;

      // Continue until 5 clusters have been built or there is no cluster left
      while (nclusters < n_clusters_max && build_cluster) {
        build_cluster = false;
        SimpleCaloHit centerhit;

        for (const auto& hit : ecalhits) {
          // Highest hit in good region with pt>1 and not used in any other cluster
          if (hit.isInCardAndRegion(cc, nregion) && !hit.used() && hit.pt() >= 1.0 && hit.pt() > centerhit.pt()) {
            centerhit = hit;
            build_cluster = true;
          }
        }
        if (build_cluster)
          nclusters++;

        // Use only the 5 most energetic clusters
        if (build_cluster && nclusters > 0 && nclusters <= n_clusters_max) {
          mycluster mc1;
          mc1.cpt = 0.0;
          mc1.cWeightedEta_ = 0.0;
          mc1.cWeightedPhi_ = 0.0;
          float leftlobe = 0;
          float rightlobe = 0;
          float e5x5 = 0;
          float n5x5 = 0;
          float e2x5_1 = 0;
          float n2x5_1 = 0;
          float e2x5_2 = 0;
          float n2x5_2 = 0;
          float e2x2_1 = 0;
          float n2x2_1 = 0;
          float e2x2_2 = 0;
          float n2x2_2 = 0;
          float e2x2_3 = 0;
          float n2x2_3 = 0;
          float e2x2_4 = 0;
          float n2x2_4 = 0;
          for (auto& hit : ecalhits) {
            if (hit.isInCardAndRegion(cc, nregion) && (hit.pt() > 0)) {
              if (abs(hit.dieta(centerhit)) <= 1 && hit.diphi(centerhit) > 2 && hit.diphi(centerhit) <= 7) {
                rightlobe += hit.pt();
              }
              if (abs(hit.dieta(centerhit)) <= 1 && hit.diphi(centerhit) < -2 && hit.diphi(centerhit) >= -7) {
                leftlobe += hit.pt();
              }
              if (abs(hit.dieta(centerhit)) <= 2 && abs(hit.diphi(centerhit)) <= 2) {
                e5x5 += hit.energy();
                n5x5++;
              }
              if ((hit.dieta(centerhit) == 1 or hit.dieta(centerhit) == 0) &&
                  (hit.diphi(centerhit) == 1 or hit.diphi(centerhit) == 0)) {
                e2x2_1 += hit.energy();
                n2x2_1++;
              }
              if ((hit.dieta(centerhit) == 0 or hit.dieta(centerhit) == -1) &&
                  (hit.diphi(centerhit) == 0 or hit.diphi(centerhit) == 1)) {
                e2x2_2 += hit.energy();
                n2x2_2++;
              }
              if ((hit.dieta(centerhit) == 0 or hit.dieta(centerhit) == 1) &&
                  (hit.diphi(centerhit) == 0 or hit.diphi(centerhit) == -1)) {
                e2x2_3 += hit.energy();
                n2x2_3++;
              }
              if ((hit.dieta(centerhit) == 0 or hit.dieta(centerhit) == -1) &&
                  (hit.diphi(centerhit) == 0 or hit.diphi(centerhit) == -1)) {
                e2x2_4 += hit.energy();
                n2x2_4++;
              }
              if ((hit.dieta(centerhit) == 0 or hit.dieta(centerhit) == 1) && abs(hit.diphi(centerhit)) <= 2) {
                e2x5_1 += hit.energy();
                n2x5_1++;
              }
              if ((hit.dieta(centerhit) == 0 or hit.dieta(centerhit) == -1) && abs(hit.diphi(centerhit)) <= 2) {
                e2x5_2 += hit.energy();
                n2x5_2++;
              }
            }
            if (hit.isInCardAndRegion(cc, nregion) && !hit.used() && hit.pt() > 0 && abs(hit.dieta(centerhit)) <= 1 &&
                abs(hit.diphi(centerhit)) <= 2) {
              // clusters 3x5 in etaxphi using only the hits in the corresponding card and in the corresponding 3x4 region
              hit.setUsed(true);
              mc1.cpt += hit.pt();
              mc1.cWeightedEta_ += float(hit.pt()) * float(hit.position().eta());
              mc1.cWeightedPhi_ = mc1.cWeightedPhi_ + (float(hit.pt()) * float(hit.position().phi()));
            }
          }
          if (do_brem && (rightlobe > 0.10 * mc1.cpt or leftlobe > 0.10 * mc1.cpt)) {
            for (auto& hit : ecalhits) {
              if (hit.isInCardAndRegion(cc, nregion) && hit.pt() > 0 && !hit.used()) {
                if (rightlobe > 0.10 * mc1.cpt && (leftlobe < 0.10 * mc1.cpt or rightlobe > leftlobe) &&
                    abs(hit.dieta(centerhit)) <= 1 && hit.diphi(centerhit) > 2 && hit.diphi(centerhit) <= 7) {
                  mc1.cpt += hit.pt();
                  hit.setUsed(true);
                  mc1.cbrem_ = 1;
                }
                if (leftlobe > 0.10 * mc1.cpt && (rightlobe < 0.10 * mc1.cpt or leftlobe >= rightlobe) &&
                    abs(hit.dieta(centerhit)) <= 1 && hit.diphi(centerhit) < -2 && hit.diphi(centerhit) >= -7) {
                  mc1.cpt += hit.pt();
                  hit.setUsed(true);
                  mc1.cbrem_ = 1;
                }
              }
            }
          }
          mc1.c5x5_ = e5x5;
          mc1.c2x5_ = max(e2x5_1, e2x5_2);
          mc1.c2x2_ = e2x2_1;
          if (e2x2_2 > mc1.c2x2_)
            mc1.c2x2_ = e2x2_2;
          if (e2x2_3 > mc1.c2x2_)
            mc1.c2x2_ = e2x2_3;
          if (e2x2_4 > mc1.c2x2_)
            mc1.c2x2_ = e2x2_4;
          mc1.cWeightedEta_ = mc1.cWeightedEta_ / mc1.cpt;
          mc1.cWeightedPhi_ = mc1.cWeightedPhi_ / mc1.cpt;
          mc1.ceta_ = getCrystal_etaID(centerhit.position().eta());
          mc1.cphi_ = getCrystal_phiID(centerhit.position().phi());
          mc1.crawphi_ = centerhit.position().phi();
          mc1.craweta_ = centerhit.position().eta();
          cluster_list[cc].push_back(mc1);
        }  // End if 5 clusters per region
      }    // End while to find the 5 clusters
    }      // End loop over regions to search for clusters
    std::sort(begin(cluster_list[cc]), end(cluster_list[cc]), [](mycluster a, mycluster b) { return a.cpt > b.cpt; });

    // Merge clusters from different regions
    for (unsigned int jj = 0; jj < unsigned(cluster_list[cc].size()); ++jj) {
      for (unsigned int kk = jj + 1; kk < unsigned(cluster_list[cc].size()); ++kk) {
        if (std::abs(cluster_list[cc][jj].ceta_ - cluster_list[cc][kk].ceta_) < 2 &&
            std::abs(cluster_list[cc][jj].cphi_ - cluster_list[cc][kk].cphi_) < 2) {  // Diagonal + exact neighbors
          if (cluster_list[cc][kk].cpt > cluster_list[cc][jj].cpt) {
            cluster_list[cc][kk].cpt += cluster_list[cc][jj].cpt;
            cluster_list[cc][kk].c5x5_ += cluster_list[cc][jj].c5x5_;
            cluster_list[cc][kk].c2x5_ += cluster_list[cc][jj].c2x5_;
            cluster_list[cc][jj].cpt = 0;
            cluster_list[cc][jj].c5x5_ = 0;
            cluster_list[cc][jj].c2x5_ = 0;
            cluster_list[cc][jj].c2x2_ = 0;
          } else {
            cluster_list[cc][jj].cpt += cluster_list[cc][kk].cpt;
            cluster_list[cc][jj].c5x5_ += cluster_list[cc][kk].c5x5_;
            cluster_list[cc][jj].c2x5_ += cluster_list[cc][kk].c2x5_;
            cluster_list[cc][kk].cpt = 0;
            cluster_list[cc][kk].c2x2_ = 0;
            cluster_list[cc][kk].c2x5_ = 0;
            cluster_list[cc][kk].c5x5_ = 0;
          }
        }
      }
      if (cluster_list[cc][jj].cpt > 0) {
        cluster_list[cc][jj].cpt =
            cluster_list[cc][jj].cpt *
            calib_(cluster_list[cc][jj].cpt,
                   std::abs(cluster_list[cc][jj].craweta_));  //Mark's calibration as a function of eta and pt
        cluster_list_merged[cc].push_back(cluster_list[cc][jj]);
      }
    }
    std::sort(begin(cluster_list_merged[cc]), end(cluster_list_merged[cc]), [](mycluster a, mycluster b) {
      return a.cpt > b.cpt;
    });

    // Fill cluster information in the arrays. We keep max 12 clusters (distributed between 4 links)
    for (unsigned int jj = 0; jj < unsigned(cluster_list_merged[cc].size()) && jj < n_clusters_4link; ++jj) {
      crystalID_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] =
          getCrystalIDInTower(cluster_list_merged[cc][jj].ceta_, cluster_list_merged[cc][jj].cphi_);
      towerID_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] =
          getTowerID(cluster_list_merged[cc][jj].ceta_, cluster_list_merged[cc][jj].cphi_);
      energy_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = cluster_list_merged[cc][jj].cpt;
      brem_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = cluster_list_merged[cc][jj].cbrem_;
      if (passes_ss(cluster_list_merged[cc][jj].cpt,
                    cluster_list_merged[cc][jj].c2x5_ / cluster_list_merged[cc][jj].c5x5_))
        showerShape_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = 1;
      else
        showerShape_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = 0;
      if (passes_looseTkss(cluster_list_merged[cc][jj].cpt,
                           cluster_list_merged[cc][jj].c2x5_ / cluster_list_merged[cc][jj].c5x5_))
        showerShapeLooseTk_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = 1;
      else
        showerShapeLooseTk_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = 0;
      if (passes_photon(cluster_list_merged[cc][jj].cpt,
                        cluster_list_merged[cc][jj].c2x2_ / cluster_list_merged[cc][jj].c2x5_))
        photonShowerShape_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = 1;
      else
        photonShowerShape_cluster_L1Card[jj % n_links_card][jj / n_links_card][cc] = 0;
    }

    // Loop over calo ecal hits to get the ECAL towers. Take only hits that have not been used to make clusters
    for (const auto& hit : ecalhits) {
      if (hit.isInCard(cc) && !hit.used()) {
        for (int jj = 0; jj < n_links_card; ++jj) {                                        // loop over 4 links per card
          if ((getCrystal_phiID(hit.position().phi()) / n_crystals_towerPhi) % 4 == jj) {  // Go to ID tower modulo 4
            for (int ii = 0; ii < n_towers_per_link; ++ii) {
              // Apply Mark's calibration at the same time (row of the lowest pT, as a function of eta)
              if ((getCrystal_etaID(hit.position().eta()) / n_crystals_towerEta) % n_towers_per_link == ii) {
                ECAL_tower_L1Card[jj][ii][cc] += hit.pt() * calib_(0, std::abs(hit.position().eta()));
                iEta_tower_L1Card[jj][ii][cc] = getTower_absoluteEtaID(hit.position().eta());
                iPhi_tower_L1Card[jj][ii][cc] = getTower_absolutePhiID(hit.position().phi());
              }
            }  // end of loop over eta towers
          }
        }  // end of loop over phi links
        // Make sure towers with 0 ET are initialized with proper iEta, iPhi coordinates
        static constexpr float tower_width = 0.0873;
        for (int jj = 0; jj < n_links_card; ++jj) {
          for (int ii = 0; ii < n_towers_per_link; ++ii) {
            float phi = getPhiMin_card(cc) * tower_width / n_crystals_towerPhi - M_PI + (jj + 0.5) * tower_width;
            float eta = getEtaMin_card(cc) * tower_width / n_crystals_towerEta - n_towers_per_link * tower_width +
                        (ii + 0.5) * tower_width;
            iEta_tower_L1Card[jj][ii][cc] = getTower_absoluteEtaID(eta);
            iPhi_tower_L1Card[jj][ii][cc] = getTower_absolutePhiID(phi);
          }
        }
      }  // end of check if inside card
    }    // end of loop over hits to build towers

    // Loop over hcal hits to get the HCAL towers.
    for (const auto& hit : hcalhits) {
      if (hit.isInCard(cc) && hit.pt() > 0) {
        for (int jj = 0; jj < n_links_card; ++jj) {
          if ((getCrystal_phiID(hit.position().phi()) / n_crystals_towerPhi) % n_links_card == jj) {
            for (int ii = 0; ii < n_towers_per_link; ++ii) {
              if ((getCrystal_etaID(hit.position().eta()) / n_crystals_towerEta) % n_towers_per_link == ii) {
                HCAL_tower_L1Card[jj][ii][cc] += hit.pt();
                iEta_tower_L1Card[jj][ii][cc] = getTower_absoluteEtaID(hit.position().eta());
                iPhi_tower_L1Card[jj][ii][cc] = getTower_absolutePhiID(hit.position().phi());
              }
            }  // end of loop over eta towers
          }
        }  // end of loop over phi links
      }    // end of check if inside card
    }      // end of loop over hits to build towers

    // Give back energy of not used clusters to the towers (if there are more than 12 clusters)
    for (unsigned int kk = n_clusters_4link; kk < cluster_list_merged[cc].size(); ++kk) {
      if (cluster_list_merged[cc][kk].cpt > 0) {
        ECAL_tower_L1Card[getTower_phiID(cluster_list_merged[cc][kk].cphi_)]
                         [getTower_etaID(cluster_list_merged[cc][kk].ceta_)][cc] += cluster_list_merged[cc][kk].cpt;
      }
    }
  }  //End of loop over cards

  //*********************************************************
  //******************** Do layer 2 *************************
  //*********************************************************

  // Definition of L2 outputs
  float HCAL_tower_L2Card[n_links_GCTcard][n_towers_per_link]
                         [n_GCTcards];  // 3 L2 cards send each 48 links with 17 towers
  float ECAL_tower_L2Card[n_links_GCTcard][n_towers_per_link][n_GCTcards];
  int iEta_tower_L2Card[n_links_GCTcard][n_towers_per_link][n_GCTcards];
  int iPhi_tower_L2Card[n_links_GCTcard][n_towers_per_link][n_GCTcards];
  float energy_cluster_L2Card[n_links_GCTcard][n_clusters_per_link]
                             [n_GCTcards];  // 3 L2 cards send each 48 links with 2 clusters
  float brem_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];
  int towerID_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];
  int crystalID_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];
  float isolation_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];
  float HE_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];
  int showerShape_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];
  int showerShapeLooseTk_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];
  int photonShowerShape_cluster_L2Card[n_links_GCTcard][n_clusters_per_link][n_GCTcards];

  for (int ii = 0; ii < n_links_GCTcard; ++ii) {
    for (int jj = 0; jj < n_towers_per_link; ++jj) {
      for (int ll = 0; ll < n_GCTcards; ++ll) {
        HCAL_tower_L2Card[ii][jj][ll] = 0;
        ECAL_tower_L2Card[ii][jj][ll] = 0;
        iEta_tower_L2Card[ii][jj][ll] = 0;
        iPhi_tower_L2Card[ii][jj][ll] = 0;
      }
    }
  }
  for (int ii = 0; ii < n_links_GCTcard; ++ii) {
    for (int jj = 0; jj < n_clusters_per_link; ++jj) {
      for (int ll = 0; ll < n_GCTcards; ++ll) {
        energy_cluster_L2Card[ii][jj][ll] = 0;
        brem_cluster_L2Card[ii][jj][ll] = 0;
        towerID_cluster_L2Card[ii][jj][ll] = 0;
        crystalID_cluster_L2Card[ii][jj][ll] = 0;
        isolation_cluster_L2Card[ii][jj][ll] = 0;
        HE_cluster_L2Card[ii][jj][ll] = 0;
        photonShowerShape_cluster_L2Card[ii][jj][ll] = 0;
        showerShape_cluster_L2Card[ii][jj][ll] = 0;
        showerShapeLooseTk_cluster_L2Card[ii][jj][ll] = 0;
      }
    }
  }

  // There is one list of clusters per equivalent of L1 card. We take the 8 highest pt.
  vector<mycluster> cluster_list_L2[n_towers_halfPhi];

  // Merge clusters on the phi edges
  for (int ii = 0; ii < n_borders_phi; ++ii) {  // 18 borders in phi
    for (int jj = 0; jj < n_eta_bins; ++jj) {   // 2 eta bins
      int card_left = 2 * ii + jj;
      int card_right = 2 * ii + jj + 2;
      if (card_right > 35)
        card_right = card_right - n_towers_halfPhi;
      for (int kk = 0; kk < n_clusters_4link; ++kk) {  // 12 clusters in the first card. We check the right side
        if (towerID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] > 50 &&
            crystalID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] > 19 &&
            energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] > 0) {
          for (int ll = 0; ll < n_clusters_4link; ++ll) {  // We check the 12 clusters in the card on the right
            if (towerID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] < n_towers_per_link &&
                crystalID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] < 5 &&
                std::abs(
                    5 * (towerID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right]) % n_towers_per_link +
                    crystalID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] % 5 -
                    5 * (towerID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left]) % n_towers_per_link -
                    crystalID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] % 5) < 2) {
              if (energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] >
                  energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right]) {
                energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] +=
                    energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right];
                energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] = 0;
              }  // The least energetic cluster is merged to the most energetic one
              else {
                energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] +=
                    energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left];
                energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] = 0;
              }
            }
          }
        }
      }
    }
  }

  // Bremsstrahlung corrections. Merge clusters on the phi edges depending on pT (pt more than 10 percent, dphi leq 5, deta leq 1)
  if (do_brem) {
    for (int ii = 0; ii < n_borders_phi; ++ii) {  // 18 borders in phi
      for (int jj = 0; jj < n_eta_bins; ++jj) {   // 2 eta bins
        int card_left = 2 * ii + jj;
        int card_right = 2 * ii + jj + 2;
        if (card_right > 35)
          card_right = card_right - n_towers_halfPhi;
        // 12 clusters in the first card. We check the right side crystalID_cluster_L1Card[kk%4][kk/4][card_left]
        for (int kk = 0; kk < n_clusters_4link; ++kk) {
          // if the tower is at the edge there might be brem corrections, whatever the crystal ID
          if (towerID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] > 50 &&
              energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] > 0) {
            for (int ll = 0; ll < n_clusters_4link; ++ll) {  // We check the 12 clusters in the card on the right
              //Distance of 1 max in eta
              if (towerID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] < n_towers_per_link &&
                  std::abs(5 * (towerID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right]) %
                               n_towers_per_link +
                           crystalID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] % 5 -
                           5 * (towerID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left]) %
                               n_towers_per_link -
                           crystalID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] % 5) <= 1) {
                //Distance of 5 max in phi
                if (towerID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] < n_towers_per_link &&
                    std::abs(5 + crystalID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] / 5 -
                             (crystalID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] / 5)) <= 5) {
                  if (energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] >
                          energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] &&
                      energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] >
                          0.10 * energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left]) {
                    energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] +=
                        energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right];
                    energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] = 0;
                    brem_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] = 1;
                  }
                  // The least energetic cluster is merged to the most energetic one, if its pt is at least ten percent
                  else if (energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_right] >=
                               energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_left] &&
                           energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_left] >
                               0.10 * energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_right]) {
                    energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] +=
                        energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left];
                    energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_left] = 0;
                    brem_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_right] = 1;
                  }
                }  //max distance eta
              }    //max distance phi
            }      //max distance phi
          }
        }
      }
    }
  }

  // Merge clusters on the eta edges
  for (int ii = 0; ii < n_borders_eta; ++ii) {  // 18 borders in eta
    int card_bottom = 2 * ii;
    int card_top = 2 * ii + 1;
    for (int kk = 0; kk < n_clusters_4link; ++kk) {  // 12 clusters in the first card. We check the top side
      if (towerID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] % n_towers_per_link == 16 &&
          crystalID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] % 5 == n_links_card &&
          energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] >
              0) {                                       // If there is one cluster on the right side of the first card
        for (int ll = 0; ll < n_clusters_4link; ++ll) {  // We check the card on the right
          if (std::abs(
                  5 * (towerID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] / n_towers_per_link) +
                  crystalID_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] / 5 -
                  5 * (towerID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_top] / n_towers_per_link) -
                  crystalID_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_top] / 5) < 2) {
            if (energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] >
                energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_bottom]) {
              energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] +=
                  energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_top];
              energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_top] = 0;
            } else {
              energy_cluster_L1Card[ll % n_links_card][ll / n_links_card][card_top] +=
                  energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom];
              energy_cluster_L1Card[kk % n_links_card][kk / n_links_card][card_bottom] = 0;
            }
          }
        }
      }
    }
  }

  // Regroup the new clusters per equivalent of L1 card geometry
  for (int ii = 0; ii < n_towers_halfPhi; ++ii) {
    for (int jj = 0; jj < n_clusters_4link; ++jj) {
      if (energy_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii] > 0) {
        mycluster mc1;
        mc1.cpt = energy_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii];
        mc1.cbrem_ = brem_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii];
        mc1.ctowerid_ = towerID_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii];
        mc1.ccrystalid_ = crystalID_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii];
        mc1.cshowershape_ = showerShape_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii];
        mc1.cshowershapeloosetk_ = showerShapeLooseTk_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii];
        mc1.cphotonshowershape_ = photonShowerShape_cluster_L1Card[jj % n_links_card][jj / n_links_card][ii];
        cluster_list_L2[ii].push_back(mc1);
      }
    }
    std::sort(
        begin(cluster_list_L2[ii]), end(cluster_list_L2[ii]), [](mycluster a, mycluster b) { return a.cpt > b.cpt; });
  }

  // If there are more than 8 clusters per equivalent of L1 card we need to put them back in the towers
  for (int ii = 0; ii < n_towers_halfPhi; ++ii) {
    for (unsigned int jj = n_clusters_per_L1card; jj < n_clusters_4link && jj < cluster_list_L2[ii].size(); ++jj) {
      if (cluster_list_L2[ii][jj].cpt > 0) {
        ECAL_tower_L1Card[cluster_list_L2[ii][jj].ctowerid_ / n_towers_per_link]
                         [cluster_list_L2[ii][jj].ctowerid_ % n_towers_per_link][ii] += cluster_list_L2[ii][jj].cpt;
        cluster_list_L2[ii][jj].cpt = 0;
        cluster_list_L2[ii][jj].ctowerid_ = 0;
        cluster_list_L2[ii][jj].ccrystalid_ = 0;
      }
    }
  }

  // Compute isolation (7*7 ECAL towers) and HCAL energy (5x5 HCAL towers)
  for (int ii = 0; ii < n_towers_halfPhi; ++ii) {  // Loop over the new cluster list (stored in 36x8 format)
    for (unsigned int jj = 0; jj < n_clusters_per_L1card && jj < cluster_list_L2[ii].size(); ++jj) {
      int cluster_etaOfTower_fullDetector = get_towerEta_fromCardTowerInCard(ii, cluster_list_L2[ii][jj].ctowerid_);
      int cluster_phiOfTower_fullDetector = get_towerPhi_fromCardTowerInCard(ii, cluster_list_L2[ii][jj].ctowerid_);
      float hcal_nrj = 0.0;
      float isolation = 0.0;
      int ntowers = 0;

      for (int iii = 0; iii < n_towers_halfPhi; ++iii) {  // The clusters have to be added to the isolation
        for (unsigned int jjj = 0; jjj < n_clusters_per_L1card && jjj < cluster_list_L2[iii].size(); ++jjj) {
          if (!(iii == ii && jjj == jj)) {
            int cluster2_eta = get_towerEta_fromCardTowerInCard(iii, cluster_list_L2[iii][jjj].ctowerid_);
            int cluster2_phi = get_towerPhi_fromCardTowerInCard(iii, cluster_list_L2[iii][jjj].ctowerid_);
            if (abs(cluster2_eta - cluster_etaOfTower_fullDetector) <= 2 &&
                (abs(cluster2_phi - cluster_phiOfTower_fullDetector) <= 2 or
                 abs(cluster2_phi - n_towers_Phi - cluster_phiOfTower_fullDetector) <= 2)) {
              isolation += cluster_list_L2[iii][jjj].cpt;
            }
          }
        }
      }
      for (int kk = 0; kk < n_towers_halfPhi; ++kk) {       // 36 cards
        for (int ll = 0; ll < n_links_card; ++ll) {         // 4 links per card
          for (int mm = 0; mm < n_towers_per_link; ++mm) {  // 17 towers per link
            int etaOftower_fullDetector = get_towerEta_fromCardLinkTower(kk, ll, mm);
            int phiOftower_fullDetector = get_towerPhi_fromCardLinkTower(kk, ll, mm);
            // First do ECAL
            // The towers are within 3. Needs to stitch the two phi sides together
            if (abs(etaOftower_fullDetector - cluster_etaOfTower_fullDetector) <= 2 &&
                (abs(phiOftower_fullDetector - cluster_phiOfTower_fullDetector) <= 2 or
                 abs(phiOftower_fullDetector - n_towers_Phi - cluster_phiOfTower_fullDetector) <= 2)) {
              // Remove the column outside of the L2 card:  values (0,71), (23,26), (24,21), (47,50), (48,50), (71,2)
              if (!((cluster_phiOfTower_fullDetector == 0 && phiOftower_fullDetector == 71) or
                    (cluster_phiOfTower_fullDetector == 23 && phiOftower_fullDetector == 26) or
                    (cluster_phiOfTower_fullDetector == 24 && phiOftower_fullDetector == 21) or
                    (cluster_phiOfTower_fullDetector == 47 && phiOftower_fullDetector == 50) or
                    (cluster_phiOfTower_fullDetector == 48 && phiOftower_fullDetector == 45) or
                    (cluster_phiOfTower_fullDetector == 71 && phiOftower_fullDetector == 2))) {
                isolation += ECAL_tower_L1Card[ll][mm][kk];
                ntowers++;
              }
            }
            // Now do HCAL
            // The towers are within 2. Needs to stitch the two phi sides together
            if (abs(etaOftower_fullDetector - cluster_etaOfTower_fullDetector) <= 2 &&
                (abs(phiOftower_fullDetector - cluster_phiOfTower_fullDetector) <= 2 or
                 abs(phiOftower_fullDetector - n_towers_Phi - cluster_phiOfTower_fullDetector) <= 2)) {
              hcal_nrj += HCAL_tower_L1Card[ll][mm][kk];
            }
          }
        }
      }
      // If we summed over fewer than 5*5 = 25 towers (because the cluster was near the edge), scale up the isolation sum
      int nTowersIn5x5Window = 5 * 5;
      cluster_list_L2[ii][jj].ciso_ = ((isolation) * (nTowersIn5x5Window / ntowers)) / cluster_list_L2[ii][jj].cpt;
      cluster_list_L2[ii][jj].crawIso_ = ((isolation) * (nTowersIn5x5Window / ntowers));
      cluster_list_L2[ii][jj].chovere_ = hcal_nrj / cluster_list_L2[ii][jj].cpt;
    }
  }

  //Reformat the information inside the 3 L2 cards
  //First let's fill the towers
  for (int ii = 0; ii < n_links_GCTcard; ++ii) {
    for (int jj = 0; jj < n_towers_per_link; ++jj) {
      for (int ll = 0; ll < 3; ++ll) {
        ECAL_tower_L2Card[ii][jj][ll] =
            ECAL_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll, ii)];
        HCAL_tower_L2Card[ii][jj][ll] =
            HCAL_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll, ii)];
        iEta_tower_L2Card[ii][jj][ll] =
            iEta_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll, ii)];
        iPhi_tower_L2Card[ii][jj][ll] =
            iPhi_tower_L1Card[convert_L2toL1_link(ii)][convert_L2toL1_tower(jj)][convert_L2toL1_card(ll, ii)];
      }
    }
  }

  //Second let's fill the clusters
  for (int ii = 0; ii < n_towers_halfPhi; ++ii) {  // The cluster list is still in the L1 like geometry
    for (unsigned int jj = 0; jj < unsigned(cluster_list_L2[ii].size()) && jj < n_clusters_per_L1card; ++jj) {
      crystalID_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                              [ii / n_clusters_4link] = cluster_list_L2[ii][jj].ccrystalid_;
      towerID_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                            [ii / n_clusters_4link] = cluster_list_L2[ii][jj].ctowerid_;
      energy_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                           [ii / n_clusters_4link] = cluster_list_L2[ii][jj].cpt;
      brem_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                         [ii / n_clusters_4link] = cluster_list_L2[ii][jj].cbrem_;
      isolation_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                              [ii / n_clusters_4link] = cluster_list_L2[ii][jj].ciso_;
      HE_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                       [ii / n_clusters_4link] = cluster_list_L2[ii][jj].chovere_;
      showerShape_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                                [ii / n_clusters_4link] = cluster_list_L2[ii][jj].cshowershape_;
      showerShapeLooseTk_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                                       [ii / n_clusters_4link] = cluster_list_L2[ii][jj].cshowershapeloosetk_;
      photonShowerShape_cluster_L2Card[n_links_card * (ii % n_clusters_4link) + jj % n_links_card][jj / n_links_card]
                                      [ii / n_clusters_4link] = cluster_list_L2[ii][jj].cphotonshowershape_;
    }
  }

  auto L1EGXtalClusters = std::make_unique<l1tp2::CaloCrystalClusterCollection>();
  auto L1EGammas = std::make_unique<l1t::EGammaBxCollection>();
  auto L1CaloTowers = std::make_unique<l1tp2::CaloTowerCollection>();

  // Fill the cluster collection and towers as well
  for (int ii = 0; ii < n_links_GCTcard; ++ii) {  // 48 links
    for (int ll = 0; ll < n_GCTcards; ++ll) {     // 3 cards
      // For looping over the Towers a few lines below
      for (int jj = 0; jj < 2; ++jj) {  // 2 L1EGs
        if (energy_cluster_L2Card[ii][jj][ll] > 0.45) {
          reco::Candidate::PolarLorentzVector p4calibrated(
              energy_cluster_L2Card[ii][jj][ll],
              getEta_fromL2LinkCardTowerCrystal(
                  ii, ll, towerID_cluster_L2Card[ii][jj][ll], crystalID_cluster_L2Card[ii][jj][ll]),
              getPhi_fromL2LinkCardTowerCrystal(
                  ii, ll, towerID_cluster_L2Card[ii][jj][ll], crystalID_cluster_L2Card[ii][jj][ll]),
              0.);
          SimpleCaloHit centerhit;
          bool is_iso = passes_iso(energy_cluster_L2Card[ii][jj][ll], isolation_cluster_L2Card[ii][jj][ll]);
          bool is_looseTkiso =
              passes_looseTkiso(energy_cluster_L2Card[ii][jj][ll], isolation_cluster_L2Card[ii][jj][ll]);
          bool is_ss = (showerShape_cluster_L2Card[ii][jj][ll] == 1);
          bool is_photon = (photonShowerShape_cluster_L2Card[ii][jj][ll] == 1) && is_ss && is_iso;
          bool is_looseTkss = (showerShapeLooseTk_cluster_L2Card[ii][jj][ll] == 1);
          // All the ID set to Standalone WP! Some dummy values for non calculated variables
          l1tp2::CaloCrystalCluster cluster(p4calibrated,
                                            energy_cluster_L2Card[ii][jj][ll],
                                            HE_cluster_L2Card[ii][jj][ll],
                                            isolation_cluster_L2Card[ii][jj][ll],
                                            centerhit.id(),
                                            -1000,
                                            float(brem_cluster_L2Card[ii][jj][ll]),
                                            -1000,
                                            -1000,
                                            energy_cluster_L2Card[ii][jj][ll],
                                            -1,
                                            is_iso && is_ss,
                                            is_iso && is_ss,
                                            is_photon,
                                            is_iso && is_ss,
                                            is_looseTkiso && is_looseTkss,
                                            is_iso && is_ss);
          // Experimental parameters, don't want to bother with hardcoding them in data format
          std::map<std::string, float> params;
          params["standaloneWP_showerShape"] = is_ss;
          params["standaloneWP_isolation"] = is_iso;
          params["trkMatchWP_showerShape"] = is_looseTkss;
          params["trkMatchWP_isolation"] = is_looseTkiso;
          params["TTiEta"] = getToweriEta_fromAbsoluteID(getTower_absoluteEtaID(p4calibrated.eta()));
          params["TTiPhi"] = getToweriPhi_fromAbsoluteID(getTower_absolutePhiID(p4calibrated.phi()));
          cluster.setExperimentalParams(params);
          L1EGXtalClusters->push_back(cluster);

          int standaloneWP = (int)(is_iso && is_ss);
          int looseL1TkMatchWP = (int)(is_looseTkiso && is_looseTkss);
          int photonWP = (int)(is_photon);
          int quality =
              (standaloneWP * std::pow(2, 0)) + (looseL1TkMatchWP * std::pow(2, 1)) + (photonWP * std::pow(2, 2));
          L1EGammas->push_back(
              0, l1t::EGamma(p4calibrated, p4calibrated.pt(), p4calibrated.eta(), p4calibrated.phi(), quality, 1));
        }
      }
    }
  }

  for (int ii = 0; ii < n_links_GCTcard; ++ii) {  // 48 links
    for (int ll = 0; ll < n_GCTcards; ++ll) {     // 3 cards
      // Fill the tower collection
      for (int jj = 0; jj < n_towers_per_link; ++jj) {  // 17 TTs
        l1tp2::CaloTower l1CaloTower;
        l1CaloTower.setEcalTowerEt(ECAL_tower_L2Card[ii][jj][ll]);
        l1CaloTower.setHcalTowerEt(HCAL_tower_L2Card[ii][jj][ll]);
        l1CaloTower.setTowerIEta(getToweriEta_fromAbsoluteID(iEta_tower_L2Card[ii][jj][ll]));
        l1CaloTower.setTowerIPhi(getToweriPhi_fromAbsoluteID(iPhi_tower_L2Card[ii][jj][ll]));
        l1CaloTower.setTowerEta(getTowerEta_fromAbsoluteID(iEta_tower_L2Card[ii][jj][ll]));
        l1CaloTower.setTowerPhi(getTowerPhi_fromAbsoluteID(iPhi_tower_L2Card[ii][jj][ll]));
        // Some towers have incorrect eta/phi because that wasn't initialized in certain cases, fix these
        static float constexpr towerEtaUpperUnitialized = -80;
        static float constexpr towerPhiUpperUnitialized = -90;
        if (l1CaloTower.towerEta() < towerEtaUpperUnitialized && l1CaloTower.towerPhi() < towerPhiUpperUnitialized) {
          l1CaloTower.setTowerEta(l1t::CaloTools::towerEta(l1CaloTower.towerIEta()));
          l1CaloTower.setTowerPhi(l1t::CaloTools::towerPhi(l1CaloTower.towerIEta(), l1CaloTower.towerIPhi()));
        }
        l1CaloTower.setIsBarrel(true);

        // Add L1EGs if they match in iEta / iPhi
        // L1EGs are already pT ordered, we will take the ID info for the leading one, but pT as the sum
        for (const auto& l1eg : *L1EGXtalClusters) {
          if (l1eg.experimentalParam("TTiEta") != l1CaloTower.towerIEta())
            continue;
          if (l1eg.experimentalParam("TTiPhi") != l1CaloTower.towerIPhi())
            continue;

          int n_L1eg = l1CaloTower.nL1eg();
          l1CaloTower.setNL1eg(n_L1eg++);
          l1CaloTower.setL1egTowerEt(l1CaloTower.l1egTowerEt() + l1eg.pt());
          // Don't record L1EG quality info for subleading L1EG
          if (l1CaloTower.nL1eg() > 1)
            continue;
          l1CaloTower.setL1egTrkSS(l1eg.experimentalParam("trkMatchWP_showerShape"));
          l1CaloTower.setL1egTrkIso(l1eg.experimentalParam("trkMatchWP_isolation"));
          l1CaloTower.setL1egStandaloneSS(l1eg.experimentalParam("standaloneWP_showerShape"));
          l1CaloTower.setL1egStandaloneIso(l1eg.experimentalParam("standaloneWP_isolation"));
        }

        L1CaloTowers->push_back(l1CaloTower);
      }
    }
  }

  iEvent.put(std::move(L1EGXtalClusters));
  iEvent.put(std::move(L1EGammas));
  iEvent.put(std::move(L1CaloTowers), "L1CaloTowerCollection");
}

bool L1EGCrystalClusterEmulatorProducer::passes_iso(float pt, float iso) {
  bool is_iso = true;
  if (pt < slideIsoPtThreshold) {
    if (!((a0_80 - a1_80 * pt) > iso))
      is_iso = false;
  } else {
    if (iso > a0)
      is_iso = false;
  }
  if (pt > plateau_ss)
    is_iso = true;
  return is_iso;
}

bool L1EGCrystalClusterEmulatorProducer::passes_looseTkiso(float pt, float iso) {
  bool is_iso = (b0 + b1 * std::exp(-b2 * pt) > iso);
  if (pt > plateau_ss)
    is_iso = true;
  return is_iso;
}

bool L1EGCrystalClusterEmulatorProducer::passes_ss(float pt, float ss) {
  bool is_ss = ((c0_ss + c1_ss * std::exp(-c2_ss * pt)) <= ss);
  if (pt > plateau_ss)
    is_ss = true;
  return is_ss;
}

bool L1EGCrystalClusterEmulatorProducer::passes_photon(float pt, float pss) {
  bool is_ss = (pss > d0 - d1 * pt);
  if (pt > plateau_ss)
    is_ss = true;
  return is_ss;
}

bool L1EGCrystalClusterEmulatorProducer::passes_looseTkss(float pt, float ss) {
  bool is_ss = ((e0_looseTkss - e1_looseTkss * std::exp(-e2_looseTkss * pt)) <= ss);
  if (pt > plateau_ss)
    is_ss = true;
  return is_ss;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1EGCrystalClusterEmulatorProducer);
