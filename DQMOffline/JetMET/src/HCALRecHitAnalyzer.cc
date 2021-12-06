#include "DQMOffline/JetMET/interface/HCALRecHitAnalyzer.h"
// author: Bobby Scurlock, University of Florida
// first version 12/7/2006
// modified: Mike Schmitt
// date: 03.05.2007
// note: 1) code rewrite. 2.) changed to loop over all hcal detids;
//       not only those within calotowers.

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//#include "PluginManager/ModuleDef.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
//#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <memory>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <TLorentzVector.h>
#include "DQMServices/Core/interface/DQMStore.h"

#define DEBUG(X)                   \
  {                                \
    if (debug_) {                  \
      std::cout << X << std::endl; \
    }                              \
  }

HCALRecHitAnalyzer::HCALRecHitAnalyzer(const edm::ParameterSet& iConfig) {
  // Retrieve Information from the Configuration File
  hBHERecHitsLabel_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("HBHERecHitsLabel"));
  hORecHitsLabel_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("HORecHitsLabel"));
  hFRecHitsLabel_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("HFRecHitsLabel"));
  caloGeomToken_ = esConsumes<edm::Transition::BeginRun>();
  debug_ = iConfig.getParameter<bool>("Debug");
  finebinning_ = iConfig.getUntrackedParameter<bool>("FineBinning");
  FolderName_ = iConfig.getUntrackedParameter<std::string>("FolderName");
}

void HCALRecHitAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  Nevents = 0;
  FillGeometry(iSetup);
}

void HCALRecHitAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {
  // get ahold of back-end interface
  //  dbe_ = edm::Service<DQMStore>().operator->();

  ibooker.setCurrentFolder(FolderName_ + "/geometry");
  hHCAL_ieta_iphi_HBMap = ibooker.book2D("METTask_HCAL_ieta_iphi_HBMap", "", 83, -41, 42, 72, 1, 73);
  hHCAL_ieta_iphi_HEMap = ibooker.book2D("METTask_HCAL_ieta_iphi_HEMap", "", 83, -41, 42, 72, 1, 73);
  hHCAL_ieta_iphi_HFMap = ibooker.book2D("METTask_HCAL_ieta_iphi_HFMap", "", 83, -41, 42, 72, 1, 73);
  hHCAL_ieta_iphi_HOMap = ibooker.book2D("METTask_HCAL_ieta_iphi_HOMap", "", 83, -41, 42, 72, 1, 73);
  hHCAL_ieta_iphi_etaMap = ibooker.book2D("METTask_HCAL_ieta_iphi_etaMap", "", 83, -41, 42, 72, 1, 73);
  hHCAL_ieta_iphi_phiMap = ibooker.book2D("METTask_HCAL_ieta_iphi_phiMap", "", 83, -41, 42, 72, 1, 73);
  hHCAL_ieta_detaMap = ibooker.book1D("METTask_HCAL_ieta_detaMap", "", 83, -41, 42);
  hHCAL_ieta_dphiMap = ibooker.book1D("METTask_HCAL_ieta_dphiMap", "", 83, -41, 42);

  // Initialize bins for geometry to -999 because z = 0 is a valid entry
  for (int i = 1; i <= 83; i++) {
    hHCAL_ieta_detaMap->setBinContent(i, -999);
    hHCAL_ieta_dphiMap->setBinContent(i, -999);

    for (int j = 1; j <= 72; j++) {
      hHCAL_ieta_iphi_HBMap->setBinContent(i, j, 0);
      hHCAL_ieta_iphi_HEMap->setBinContent(i, j, 0);
      hHCAL_ieta_iphi_HFMap->setBinContent(i, j, 0);
      hHCAL_ieta_iphi_HOMap->setBinContent(i, j, 0);
      hHCAL_ieta_iphi_etaMap->setBinContent(i, j, -999);
      hHCAL_ieta_iphi_phiMap->setBinContent(i, j, -999);
    }
  }

  //    ibooker.setCurrentFolder("RecoMETV/MET_HCAL/data");
  ibooker.setCurrentFolder(FolderName_);
  //--Store number of events used
  hHCAL_Nevents = ibooker.book1D("METTask_HCAL_Nevents", "", 1, 0, 1);
  //--Data integrated over all events and stored by HCAL(ieta,iphi)

  hHCAL_D1_energy_ieta_iphi = ibooker.book2D("METTask_HCAL_D1_energy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D2_energy_ieta_iphi = ibooker.book2D("METTask_HCAL_D2_energy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D3_energy_ieta_iphi = ibooker.book2D("METTask_HCAL_D3_energy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D4_energy_ieta_iphi = ibooker.book2D("METTask_HCAL_D4_energy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);

  hHCAL_D1_Minenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D1_Minenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D2_Minenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D2_Minenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D3_Minenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D3_Minenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D4_Minenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D4_Minenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);

  hHCAL_D1_Maxenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D1_Maxenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D2_Maxenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D2_Maxenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D3_Maxenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D3_Maxenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D4_Maxenergy_ieta_iphi = ibooker.book2D("METTask_HCAL_D4_Maxenergy_ieta_iphi", "", 83, -41, 42, 72, 1, 73);

  // need to initialize those
  for (int i = 1; i <= 83; i++)
    for (int j = 1; j <= 73; j++) {
      hHCAL_D1_Maxenergy_ieta_iphi->setBinContent(i, j, -999);
      hHCAL_D2_Maxenergy_ieta_iphi->setBinContent(i, j, -999);
      hHCAL_D3_Maxenergy_ieta_iphi->setBinContent(i, j, -999);
      hHCAL_D4_Maxenergy_ieta_iphi->setBinContent(i, j, -999);

      hHCAL_D1_Minenergy_ieta_iphi->setBinContent(i, j, 14000);
      hHCAL_D2_Minenergy_ieta_iphi->setBinContent(i, j, 14000);
      hHCAL_D3_Minenergy_ieta_iphi->setBinContent(i, j, 14000);
      hHCAL_D4_Minenergy_ieta_iphi->setBinContent(i, j, 14000);
    }

  hHCAL_D1_Occ_ieta_iphi = ibooker.book2D("METTask_HCAL_D1_Occ_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D2_Occ_ieta_iphi = ibooker.book2D("METTask_HCAL_D2_Occ_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D3_Occ_ieta_iphi = ibooker.book2D("METTask_HCAL_D3_Occ_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  hHCAL_D4_Occ_ieta_iphi = ibooker.book2D("METTask_HCAL_D4_Occ_ieta_iphi", "", 83, -41, 42, 72, 1, 73);
  //--Data over eta-rings

  // CaloTower values

  if (finebinning_) {
    hHCAL_D1_energyvsieta = ibooker.book2D("METTask_HCAL_D1_energyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D2_energyvsieta = ibooker.book2D("METTask_HCAL_D2_energyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D3_energyvsieta = ibooker.book2D("METTask_HCAL_D3_energyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D4_energyvsieta = ibooker.book2D("METTask_HCAL_D4_energyvsieta", "", 83, -41, 42, 20101, -100, 2001);

    hHCAL_D1_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D1_Minenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D2_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D2_Minenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D3_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D3_Minenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D4_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D4_Minenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);

    hHCAL_D1_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D1_Maxenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D2_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D2_Maxenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D3_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D3_Maxenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);
    hHCAL_D4_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D4_Maxenergyvsieta", "", 83, -41, 42, 20101, -100, 2001);

    // Integrated over phi
    hHCAL_D1_Occvsieta = ibooker.book2D("METTask_HCAL_D1_Occvsieta", "", 83, -41, 42, 73, 0, 73);
    hHCAL_D2_Occvsieta = ibooker.book2D("METTask_HCAL_D2_Occvsieta", "", 83, -41, 42, 73, 0, 73);
    hHCAL_D3_Occvsieta = ibooker.book2D("METTask_HCAL_D3_Occvsieta", "", 83, -41, 42, 73, 0, 73);
    hHCAL_D4_Occvsieta = ibooker.book2D("METTask_HCAL_D4_Occvsieta", "", 83, -41, 42, 73, 0, 73);

    hHCAL_D1_SETvsieta = ibooker.book2D("METTask_HCAL_D1_SETvsieta", "", 83, -41, 42, 20001, 0, 2001);
    hHCAL_D2_SETvsieta = ibooker.book2D("METTask_HCAL_D2_SETvsieta", "", 83, -41, 42, 20001, 0, 2001);
    hHCAL_D3_SETvsieta = ibooker.book2D("METTask_HCAL_D3_SETvsieta", "", 83, -41, 42, 20001, 0, 2001);
    hHCAL_D4_SETvsieta = ibooker.book2D("METTask_HCAL_D4_SETvsieta", "", 83, -41, 42, 20001, 0, 2001);

    hHCAL_D1_METvsieta = ibooker.book2D("METTask_HCAL_D1_METvsieta", "", 83, -41, 42, 20001, 0, 2001);
    hHCAL_D2_METvsieta = ibooker.book2D("METTask_HCAL_D2_METvsieta", "", 83, -41, 42, 20001, 0, 2001);
    hHCAL_D3_METvsieta = ibooker.book2D("METTask_HCAL_D3_METvsieta", "", 83, -41, 42, 20001, 0, 2001);
    hHCAL_D4_METvsieta = ibooker.book2D("METTask_HCAL_D4_METvsieta", "", 83, -41, 42, 20001, 0, 2001);

    hHCAL_D1_METPhivsieta = ibooker.book2D("METTask_HCAL_D1_METPhivsieta", "", 83, -41, 42, 80, -4, 4);
    hHCAL_D2_METPhivsieta = ibooker.book2D("METTask_HCAL_D2_METPhivsieta", "", 83, -41, 42, 80, -4, 4);
    hHCAL_D3_METPhivsieta = ibooker.book2D("METTask_HCAL_D3_METPhivsieta", "", 83, -41, 42, 80, -4, 4);
    hHCAL_D4_METPhivsieta = ibooker.book2D("METTask_HCAL_D4_METPhivsieta", "", 83, -41, 42, 80, -4, 4);

    hHCAL_D1_MExvsieta = ibooker.book2D("METTask_HCAL_D1_MExvsieta", "", 83, -41, 42, 10001, -500, 501);
    hHCAL_D2_MExvsieta = ibooker.book2D("METTask_HCAL_D2_MExvsieta", "", 83, -41, 42, 10001, -500, 501);
    hHCAL_D3_MExvsieta = ibooker.book2D("METTask_HCAL_D3_MExvsieta", "", 83, -41, 42, 10001, -500, 501);
    hHCAL_D4_MExvsieta = ibooker.book2D("METTask_HCAL_D4_MExvsieta", "", 83, -41, 42, 10001, -500, 501);

    hHCAL_D1_MEyvsieta = ibooker.book2D("METTask_HCAL_D1_MEyvsieta", "", 83, -41, 42, 10001, -500, 501);
    hHCAL_D2_MEyvsieta = ibooker.book2D("METTask_HCAL_D2_MEyvsieta", "", 83, -41, 42, 10001, -500, 501);
    hHCAL_D3_MEyvsieta = ibooker.book2D("METTask_HCAL_D3_MEyvsieta", "", 83, -41, 42, 10001, -500, 501);
    hHCAL_D4_MEyvsieta = ibooker.book2D("METTask_HCAL_D4_MEyvsieta", "", 83, -41, 42, 10001, -500, 501);
  } else {
    hHCAL_D1_energyvsieta = ibooker.book2D("METTask_HCAL_D1_energyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D2_energyvsieta = ibooker.book2D("METTask_HCAL_D2_energyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D3_energyvsieta = ibooker.book2D("METTask_HCAL_D3_energyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D4_energyvsieta = ibooker.book2D("METTask_HCAL_D4_energyvsieta", "", 83, -41, 42, 1000, -10, 1990);

    hHCAL_D1_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D1_Minenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D2_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D2_Minenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D3_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D3_Minenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D4_Minenergyvsieta = ibooker.book2D("METTask_HCAL_D4_Minenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);

    hHCAL_D1_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D1_Maxenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D2_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D2_Maxenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D3_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D3_Maxenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);
    hHCAL_D4_Maxenergyvsieta = ibooker.book2D("METTask_HCAL_D4_Maxenergyvsieta", "", 83, -41, 42, 1000, -10, 1990);

    // Integrated over phi
    hHCAL_D1_Occvsieta = ibooker.book2D("METTask_HCAL_D1_Occvsieta", "", 83, -41, 42, 73, 0, 73);
    hHCAL_D2_Occvsieta = ibooker.book2D("METTask_HCAL_D2_Occvsieta", "", 83, -41, 42, 73, 0, 73);
    hHCAL_D3_Occvsieta = ibooker.book2D("METTask_HCAL_D3_Occvsieta", "", 83, -41, 42, 73, 0, 73);
    hHCAL_D4_Occvsieta = ibooker.book2D("METTask_HCAL_D4_Occvsieta", "", 83, -41, 42, 73, 0, 73);

    hHCAL_D1_SETvsieta = ibooker.book2D("METTask_HCAL_D1_SETvsieta", "", 83, -41, 42, 1000, 0, 2000);
    hHCAL_D2_SETvsieta = ibooker.book2D("METTask_HCAL_D2_SETvsieta", "", 83, -41, 42, 1000, 0, 2000);
    hHCAL_D3_SETvsieta = ibooker.book2D("METTask_HCAL_D3_SETvsieta", "", 83, -41, 42, 1000, 0, 2000);
    hHCAL_D4_SETvsieta = ibooker.book2D("METTask_HCAL_D4_SETvsieta", "", 83, -41, 42, 1000, 0, 2000);

    hHCAL_D1_METvsieta = ibooker.book2D("METTask_HCAL_D1_METvsieta", "", 83, -41, 42, 1000, 0, 2000);
    hHCAL_D2_METvsieta = ibooker.book2D("METTask_HCAL_D2_METvsieta", "", 83, -41, 42, 1000, 0, 2000);
    hHCAL_D3_METvsieta = ibooker.book2D("METTask_HCAL_D3_METvsieta", "", 83, -41, 42, 1000, 0, 2000);
    hHCAL_D4_METvsieta = ibooker.book2D("METTask_HCAL_D4_METvsieta", "", 83, -41, 42, 1000, 0, 2000);

    hHCAL_D1_METPhivsieta = ibooker.book2D("METTask_HCAL_D1_METPhivsieta", "", 83, -41, 42, 80, -4, 4);
    hHCAL_D2_METPhivsieta = ibooker.book2D("METTask_HCAL_D2_METPhivsieta", "", 83, -41, 42, 80, -4, 4);
    hHCAL_D3_METPhivsieta = ibooker.book2D("METTask_HCAL_D3_METPhivsieta", "", 83, -41, 42, 80, -4, 4);
    hHCAL_D4_METPhivsieta = ibooker.book2D("METTask_HCAL_D4_METPhivsieta", "", 83, -41, 42, 80, -4, 4);

    hHCAL_D1_MExvsieta = ibooker.book2D("METTask_HCAL_D1_MExvsieta", "", 83, -41, 42, 500, -500, 500);
    hHCAL_D2_MExvsieta = ibooker.book2D("METTask_HCAL_D2_MExvsieta", "", 83, -41, 42, 500, -500, 500);
    hHCAL_D3_MExvsieta = ibooker.book2D("METTask_HCAL_D3_MExvsieta", "", 83, -41, 42, 500, -500, 500);
    hHCAL_D4_MExvsieta = ibooker.book2D("METTask_HCAL_D4_MExvsieta", "", 83, -41, 42, 500, -500, 500);

    hHCAL_D1_MEyvsieta = ibooker.book2D("METTask_HCAL_D1_MEyvsieta", "", 83, -41, 42, 500, -500, 500);
    hHCAL_D2_MEyvsieta = ibooker.book2D("METTask_HCAL_D2_MEyvsieta", "", 83, -41, 42, 500, -500, 500);
    hHCAL_D3_MEyvsieta = ibooker.book2D("METTask_HCAL_D3_MEyvsieta", "", 83, -41, 42, 500, -500, 500);
    hHCAL_D4_MEyvsieta = ibooker.book2D("METTask_HCAL_D4_MEyvsieta", "", 83, -41, 42, 500, -500, 500);
  }
  // Inspect Setup for CaloTower Geometry
  //  FillGeometry(iSetup);
}

void HCALRecHitAnalyzer::FillGeometry(const edm::EventSetup& iSetup) {
  // ==========================================================
  // Retrieve!
  // ==========================================================

  const auto& pG = iSetup.getHandle(caloGeomToken_);

  if (!pG.isValid()) {
    edm::LogInfo("OutputInfo") << "Failed to retrieve an Event Setup Handle, Aborting Task "
                               << "HCALRecHitAnalyzer::FillGeometry!\n";
    return;
  }

  const CaloGeometry cG = *pG;

  const HcalGeometry* HBgeom = dynamic_cast<const HcalGeometry*>(cG.getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  const HcalGeometry* HEgeom = dynamic_cast<const HcalGeometry*>(cG.getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
  const CaloSubdetectorGeometry* HOgeom = cG.getSubdetectorGeometry(DetId::Hcal, HcalOuter);
  const CaloSubdetectorGeometry* HFgeom = cG.getSubdetectorGeometry(DetId::Hcal, HcalForward);

  // ==========================================================
  // Fill Histograms!
  // ==========================================================

  std::vector<DetId>::iterator i;

  int HBmin_ieta = 99, HBmax_ieta = -99;
  int HBmin_iphi = 99, HBmax_iphi = -99;

  // Loop Over all Hcal Barrel DetId's
  int nHBdetid = 0;
  std::vector<DetId> HBids = HBgeom->getValidDetIds(DetId::Hcal, HcalBarrel);

  for (i = HBids.begin(); i != HBids.end(); i++) {
    nHBdetid++;

    HcalDetId HcalID(*i);

    int Calo_ieta = 42 + HcalID.ieta();
    int Calo_iphi = HcalID.iphi();
    double Calo_eta = HBgeom->getPosition(HcalID).eta();
    double Calo_phi = HBgeom->getPosition(HcalID).phi();

    if (hHCAL_ieta_iphi_etaMap->getBinContent(Calo_ieta, Calo_iphi) == -999) {
      hHCAL_ieta_iphi_etaMap->setBinContent(Calo_ieta, Calo_iphi, Calo_eta);
      hHCAL_ieta_iphi_phiMap->setBinContent(Calo_ieta, Calo_iphi, Calo_phi * 180.0 / M_PI);
    }

    if (Calo_ieta > HBmax_ieta)
      HBmax_ieta = Calo_ieta;
    if (Calo_ieta < HBmin_ieta)
      HBmin_ieta = Calo_ieta;
    if (Calo_iphi > HBmax_iphi)
      HBmax_iphi = Calo_iphi;
    if (Calo_iphi > HBmax_iphi)
      HBmin_iphi = Calo_iphi;
  }

  int HEmin_ieta = 99, HEmax_ieta = -99;
  int HEmin_iphi = 99, HEmax_iphi = -99;

  // Loop Over all Hcal Endcap DetId's
  int nHEdetid = 0;
  std::vector<DetId> HEids = HEgeom->getValidDetIds(DetId::Hcal, HcalEndcap);

  for (i = HEids.begin(); i != HEids.end(); i++) {
    nHEdetid++;

    HcalDetId HcalID(*i);

    int Calo_ieta = 42 + HcalID.ieta();
    int Calo_iphi = HcalID.iphi();
    double Calo_eta = HEgeom->getPosition(HcalID).eta();
    double Calo_phi = HEgeom->getPosition(HcalID).phi();

    // HCAL to HE eta, phi map comparison
    if (hHCAL_ieta_iphi_etaMap->getBinContent(Calo_ieta, Calo_iphi) == -999) {
      hHCAL_ieta_iphi_etaMap->setBinContent(Calo_ieta, Calo_iphi, Calo_eta);
      hHCAL_ieta_iphi_phiMap->setBinContent(Calo_ieta, Calo_iphi, Calo_phi * 180.0 / M_PI);
    }

    if (Calo_ieta > HEmax_ieta)
      HEmax_ieta = Calo_ieta;
    if (Calo_ieta < HEmin_ieta)
      HEmin_ieta = Calo_ieta;
    if (Calo_iphi > HEmax_iphi)
      HEmax_iphi = Calo_iphi;
    if (Calo_iphi > HEmax_iphi)
      HEmin_iphi = Calo_iphi;
  }

  int HFmin_ieta = 99, HFmax_ieta = -99;
  int HFmin_iphi = 99, HFmax_iphi = -99;

  // Loop Over all Hcal Forward DetId's
  int nHFdetid = 0;
  std::vector<DetId> HFids = HFgeom->getValidDetIds(DetId::Hcal, HcalForward);

  for (i = HFids.begin(); i != HFids.end(); i++) {
    nHFdetid++;

    auto cell = HFgeom->getGeometry(*i);
    HcalDetId HcalID(i->rawId());
    //GlobalPoint p = cell->getPosition();

    int Calo_ieta = 42 + HcalID.ieta();
    int Calo_iphi = HcalID.iphi();
    double Calo_eta = cell->getPosition().eta();
    double Calo_phi = cell->getPosition().phi();

    // HCAL to HF eta, phi map comparison
    if (hHCAL_ieta_iphi_etaMap->getBinContent(Calo_ieta, Calo_iphi) == -999) {
      hHCAL_ieta_iphi_etaMap->setBinContent(Calo_ieta, Calo_iphi, Calo_eta);
      hHCAL_ieta_iphi_phiMap->setBinContent(Calo_ieta, Calo_iphi, Calo_phi * 180.0 / M_PI);
    }

    if (Calo_ieta > HFmax_ieta)
      HFmax_ieta = Calo_ieta;
    if (Calo_ieta < HFmin_ieta)
      HFmin_ieta = Calo_ieta;
    if (Calo_iphi > HFmax_iphi)
      HFmax_iphi = Calo_iphi;
    if (Calo_iphi > HFmax_iphi)
      HFmin_iphi = Calo_iphi;
  }

  int HOmin_ieta = 99, HOmax_ieta = -99;
  int HOmin_iphi = 99, HOmax_iphi = -99;

  // Loop Over all Hcal Outer DetId's
  int nHOdetid = 0;
  std::vector<DetId> HOids = HOgeom->getValidDetIds(DetId::Hcal, HcalOuter);

  for (i = HOids.begin(); i != HOids.end(); i++) {
    nHOdetid++;

    auto cell = HOgeom->getGeometry(*i);
    HcalDetId HcalID(i->rawId());
    //GlobalPoint p = cell->getPosition();

    int Calo_ieta = 42 + HcalID.ieta();
    int Calo_iphi = HcalID.iphi();
    double Calo_eta = cell->getPosition().eta();
    double Calo_phi = cell->getPosition().phi();

    // HCAL to HO eta, phi map comparison
    if (hHCAL_ieta_iphi_etaMap->getBinContent(Calo_ieta, Calo_iphi) == -999) {
      hHCAL_ieta_iphi_etaMap->setBinContent(Calo_ieta, Calo_iphi, Calo_eta);
      hHCAL_ieta_iphi_phiMap->setBinContent(Calo_ieta, Calo_iphi, Calo_phi * 180.0 / M_PI);
    }

    if (Calo_ieta > HOmax_ieta)
      HOmax_ieta = Calo_ieta;
    if (Calo_ieta < HOmin_ieta)
      HOmin_ieta = Calo_ieta;
    if (Calo_iphi > HOmax_iphi)
      HOmax_iphi = Calo_iphi;
    if (Calo_iphi > HOmax_iphi)
      HOmin_iphi = Calo_iphi;
  }

  // Set the Cell Size for each (ieta, iphi) Bin
  double currentLowEdge_eta = 0;  //double currentHighEdge_eta = 0;
  for (int ieta = 1; ieta < 42; ieta++) {
    int ieta_ = 42 + ieta;
    double eta = hHCAL_ieta_iphi_etaMap->getBinContent(ieta_, 3);
    double phi = hHCAL_ieta_iphi_phiMap->getBinContent(ieta_, 3);
    double deta = 2.0 * (eta - currentLowEdge_eta);
    deta = ((float)((int)(1.0E3 * deta + 0.5))) / 1.0E3;
    double dphi = 2.0 * phi;
    if (ieta == 40 || ieta == 41)
      dphi = 20;
    if (ieta <= 39 && ieta >= 21)
      dphi = 10;
    if (ieta <= 20)
      dphi = 5;
    // BS: This is WRONG...need to correct overlap
    if (ieta == 28)
      deta = 0.218;
    if (ieta == 29)
      deta = 0.096;
    currentLowEdge_eta += deta;

    // BS: This is WRONG...need to correct overlap
    if (ieta == 29)
      currentLowEdge_eta = 2.964;

    hHCAL_ieta_detaMap->setBinContent(ieta_, deta);      // positive rings
    hHCAL_ieta_dphiMap->setBinContent(ieta_, dphi);      // positive rings
    hHCAL_ieta_detaMap->setBinContent(42 - ieta, deta);  // negative rings
    hHCAL_ieta_dphiMap->setBinContent(42 - ieta, dphi);  // negative rings

  }  // end loop over ieta

  edm::LogInfo("OutputInfo") << "HB ieta range: " << HBmin_ieta << " " << HBmax_ieta;
  edm::LogInfo("OutputInfo") << "HB iphi range: " << HBmin_iphi << " " << HBmax_iphi;
  edm::LogInfo("OutputInfo") << "HE ieta range: " << HEmin_ieta << " " << HEmax_ieta;
  edm::LogInfo("OutputInfo") << "HE iphi range: " << HEmin_iphi << " " << HEmax_iphi;
  edm::LogInfo("OutputInfo") << "HF ieta range: " << HFmin_ieta << " " << HFmax_ieta;
  edm::LogInfo("OutputInfo") << "HF iphi range: " << HFmin_iphi << " " << HFmax_iphi;
  edm::LogInfo("OutputInfo") << "HO ieta range: " << HOmin_ieta << " " << HOmax_ieta;
  edm::LogInfo("OutputInfo") << "HO iphi range: " << HOmin_iphi << " " << HOmax_iphi;
}

void HCALRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  Nevents++;
  hHCAL_Nevents->Fill(0);
  // ==========================================================
  // Retrieve!
  // ==========================================================

  const HBHERecHitCollection* HBHERecHits;
  const HORecHitCollection* HORecHits;
  const HFRecHitCollection* HFRecHits;

  edm::Handle<HBHERecHitCollection> HBHERecHitsHandle;
  iEvent.getByToken(hBHERecHitsLabel_, HBHERecHitsHandle);
  if (!HBHERecHitsHandle.isValid()) {
    edm::LogInfo("OutputInfo") << "Failed to retrieve an Event Handle, Aborting Task "
                               << "HCALRecHitAnalyzer::analyze!\n";
    return;
  } else {
    HBHERecHits = HBHERecHitsHandle.product();
  }
  edm::Handle<HORecHitCollection> HORecHitsHandle;
  iEvent.getByToken(hORecHitsLabel_, HORecHitsHandle);
  if (!HORecHitsHandle.isValid()) {
    edm::LogInfo("OutputInfo") << "Failed to retrieve an Event Handle, Aborting Task "
                               << "HCALRecHitAnalyzer::analyze!\n";
    return;
  } else {
    HORecHits = HORecHitsHandle.product();
  }
  edm::Handle<HFRecHitCollection> HFRecHitsHandle;
  iEvent.getByToken(hFRecHitsLabel_, HFRecHitsHandle);
  if (!HFRecHitsHandle.isValid()) {
    edm::LogInfo("OutputInfo") << "Failed to retrieve an Event Handle, Aborting Task "
                               << "HCALRecHitAnalyzer::analyze!\n";
    return;
  } else {
    HFRecHits = HFRecHitsHandle.product();
  }

  // ==========================================================
  // Fill Histograms!
  // ==========================================================

  TLorentzVector vHBHEMET_EtaRing[83][4];
  int HBHEActiveRing[83][4];
  int HBHENActiveCells[83][4];
  double HBHESET_EtaRing[83][4];
  double HBHEMinEnergy_EtaRing[83][4];
  double HBHEMaxEnergy_EtaRing[83][4];

  for (int i = 0; i < 83; i++) {
    for (int j = 0; j < 4; j++) {
      HBHEActiveRing[i][j] = 0;
      HBHENActiveCells[i][j] = 0;
      HBHESET_EtaRing[i][j] = 0;
      HBHEMinEnergy_EtaRing[i][j] = 14E3;
      HBHEMaxEnergy_EtaRing[i][j] = -999;
    }
  }

  // Loop over HBHERecHit's
  HBHERecHitCollection::const_iterator hbherechit;
  int nHBrechit = 0, nHErechit = 0;

  for (hbherechit = HBHERecHits->begin(); hbherechit != HBHERecHits->end(); hbherechit++) {
    HcalDetId det = hbherechit->id();
    double Energy = hbherechit->energy();
    Int_t depth = det.depth();
    Int_t ieta = det.ieta();
    Int_t iphi = det.iphi();
    int EtaRing = 41 + ieta;  // this counts from 0
    double eta = hHCAL_ieta_iphi_etaMap->getBinContent(EtaRing + 1, iphi);
    double phi = hHCAL_ieta_iphi_phiMap->getBinContent(EtaRing + 1, iphi);
    double theta = 2 * TMath::ATan(exp(-1 * eta));
    double ET = Energy * TMath::Sin(theta);
    HcalSubdetector HcalNum = det.subdet();
    TLorentzVector v_;

    if (Energy > 0)  // zero suppress
    {
      HBHEActiveRing[EtaRing][depth - 1] = 1;
      HBHENActiveCells[EtaRing][depth - 1]++;
      HBHESET_EtaRing[EtaRing][depth - 1] += ET;
      v_.SetPtEtaPhiE(ET, 0, phi, ET);
      vHBHEMET_EtaRing[EtaRing][depth - 1] -= v_;

      DEBUG(EtaRing << " " << Energy << " " << ET << " " << theta << " " << eta << " " << phi << " : "
                    << vHBHEMET_EtaRing[EtaRing][depth - 1].Phi() << " " << vHBHEMET_EtaRing[EtaRing][depth - 1].Pt());

      switch (depth) {
        case 1:
          hHCAL_D1_Occ_ieta_iphi->Fill(ieta, iphi);
          break;
        case 2:
          hHCAL_D2_Occ_ieta_iphi->Fill(ieta, iphi);
          break;
        case 3:
          hHCAL_D3_Occ_ieta_iphi->Fill(ieta, iphi);
          break;
      }  // end switch
    }

    if (Energy > HBHEMaxEnergy_EtaRing[EtaRing][depth - 1])
      HBHEMaxEnergy_EtaRing[EtaRing][depth - 1] = Energy;
    if (Energy < HBHEMinEnergy_EtaRing[EtaRing][depth - 1])
      HBHEMinEnergy_EtaRing[EtaRing][depth - 1] = Energy;

    switch (depth) {
      case 1:
        hHCAL_D1_energy_ieta_iphi->Fill(ieta, iphi, Energy);
        hHCAL_D1_energyvsieta->Fill(ieta, Energy);
        if (Energy > hHCAL_D1_Maxenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D1_Maxenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        if (Energy < hHCAL_D1_Minenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D1_Minenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);

        break;
      case 2:
        hHCAL_D2_energy_ieta_iphi->Fill(ieta, iphi, Energy);

        hHCAL_D2_energyvsieta->Fill(ieta, Energy);
        if (Energy > hHCAL_D2_Maxenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D2_Maxenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        if (Energy < hHCAL_D2_Minenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D2_Minenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        break;
      case 3:
        hHCAL_D3_energy_ieta_iphi->Fill(ieta, iphi, Energy);

        hHCAL_D3_energyvsieta->Fill(ieta, Energy);
        if (Energy > hHCAL_D3_Maxenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D3_Maxenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        if (Energy < hHCAL_D3_Minenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D3_Minenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        break;
    }  // end switch

    if (HcalNum == HcalBarrel) {
      nHBrechit++;
    } else {  // HcalEndcap
      nHErechit++;
    }
  }  // end loop over HBHERecHit's

  // Fill eta-ring MET quantities
  for (int iEtaRing = 0; iEtaRing < 83; iEtaRing++) {
    for (int jDepth = 0; jDepth < 3; jDepth++) {
      switch (jDepth + 1) {
        case 1:
          hHCAL_D1_Maxenergyvsieta->Fill(iEtaRing - 41, HBHEMaxEnergy_EtaRing[iEtaRing][jDepth]);
          hHCAL_D1_Minenergyvsieta->Fill(iEtaRing - 41, HBHEMinEnergy_EtaRing[iEtaRing][jDepth]);
          break;
        case 2:
          hHCAL_D2_Maxenergyvsieta->Fill(iEtaRing - 41, HBHEMaxEnergy_EtaRing[iEtaRing][jDepth]);
          hHCAL_D2_Minenergyvsieta->Fill(iEtaRing - 41, HBHEMinEnergy_EtaRing[iEtaRing][jDepth]);
          break;
        case 3:
          hHCAL_D3_Maxenergyvsieta->Fill(iEtaRing - 41, HBHEMaxEnergy_EtaRing[iEtaRing][jDepth]);
          hHCAL_D3_Minenergyvsieta->Fill(iEtaRing - 41, HBHEMinEnergy_EtaRing[iEtaRing][jDepth]);
          break;
      }

      if (HBHEActiveRing[iEtaRing][jDepth]) {
        switch (jDepth + 1) {
          case 1:
            hHCAL_D1_METPhivsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Phi());
            hHCAL_D1_MExvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Px());
            hHCAL_D1_MEyvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Py());
            hHCAL_D1_METvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Pt());
            hHCAL_D1_SETvsieta->Fill(iEtaRing - 41, HBHESET_EtaRing[iEtaRing][jDepth]);
            hHCAL_D1_Occvsieta->Fill(iEtaRing - 41, HBHENActiveCells[iEtaRing][jDepth]);
            break;
          case 2:
            hHCAL_D2_METPhivsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Phi());
            hHCAL_D2_MExvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Px());
            hHCAL_D2_MEyvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Py());
            hHCAL_D2_METvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Pt());
            hHCAL_D2_SETvsieta->Fill(iEtaRing - 41, HBHESET_EtaRing[iEtaRing][jDepth]);
            hHCAL_D2_Occvsieta->Fill(iEtaRing - 41, HBHENActiveCells[iEtaRing][jDepth]);
            break;
          case 3:
            hHCAL_D3_METPhivsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Phi());
            hHCAL_D3_MExvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Px());
            hHCAL_D3_MEyvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Py());
            hHCAL_D3_METvsieta->Fill(iEtaRing - 41, vHBHEMET_EtaRing[iEtaRing][jDepth].Pt());
            hHCAL_D3_SETvsieta->Fill(iEtaRing - 41, HBHESET_EtaRing[iEtaRing][jDepth]);
            hHCAL_D3_Occvsieta->Fill(iEtaRing - 41, HBHENActiveCells[iEtaRing][jDepth]);
            break;
        }  //switch
      }    //activering
    }      //depth
  }        //etaring

  //-------------------------------------------------HO
  TLorentzVector vHOMET_EtaRing[83];
  int HOActiveRing[83];
  int HONActiveCells[83];
  double HOSET_EtaRing[83];
  double HOMinEnergy_EtaRing[83];
  double HOMaxEnergy_EtaRing[83];

  for (int i = 0; i < 83; i++) {
    HOActiveRing[i] = 0;
    HONActiveCells[i] = 0;
    HOSET_EtaRing[i] = 0;
    HOMinEnergy_EtaRing[i] = 14E3;
    HOMaxEnergy_EtaRing[i] = -999;
  }

  // Loop over HORecHit's
  HORecHitCollection::const_iterator horechit;
  int nHOrechit = 0;

  for (horechit = HORecHits->begin(); horechit != HORecHits->end(); horechit++) {
    nHOrechit++;

    HcalDetId det = horechit->id();
    double Energy = horechit->energy();
    ///Int_t depth = det.depth(); //always 4
    Int_t ieta = det.ieta();
    Int_t iphi = det.iphi();
    int EtaRing = 41 + ieta;  // this counts from 0
    double eta = hHCAL_ieta_iphi_etaMap->getBinContent(EtaRing + 1, iphi);
    double phi = hHCAL_ieta_iphi_phiMap->getBinContent(EtaRing + 1, iphi);
    double theta = 2 * TMath::ATan(exp(-1 * eta));
    double ET = Energy * TMath::Sin(theta);
    TLorentzVector v_;

    if (Energy > 0)  // zero suppress
    {
      HOActiveRing[EtaRing] = 1;
      HONActiveCells[EtaRing]++;
      HOSET_EtaRing[EtaRing] += ET;
      v_.SetPtEtaPhiE(ET, 0, phi, ET);
      vHOMET_EtaRing[EtaRing] -= v_;

      hHCAL_D4_Occ_ieta_iphi->Fill(ieta, iphi);
    }

    if (Energy > HOMaxEnergy_EtaRing[EtaRing])
      HOMaxEnergy_EtaRing[EtaRing] = Energy;
    if (Energy < HOMinEnergy_EtaRing[EtaRing])
      HOMinEnergy_EtaRing[EtaRing] = Energy;

    hHCAL_D4_energy_ieta_iphi->Fill(ieta, iphi, Energy);

    hHCAL_D4_energyvsieta->Fill(ieta, Energy);
    if (Energy > hHCAL_D4_Maxenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
      hHCAL_D4_Maxenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
    if (Energy < hHCAL_D4_Minenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
      hHCAL_D4_Minenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);

  }  // end loop over HORecHit's

  // Fill eta-ring MET quantities
  for (int iEtaRing = 0; iEtaRing < 83; iEtaRing++) {
    hHCAL_D4_Maxenergyvsieta->Fill(iEtaRing - 41, HOMaxEnergy_EtaRing[iEtaRing]);
    hHCAL_D4_Minenergyvsieta->Fill(iEtaRing - 41, HOMinEnergy_EtaRing[iEtaRing]);

    if (HOActiveRing[iEtaRing]) {
      hHCAL_D4_METPhivsieta->Fill(iEtaRing - 41, vHOMET_EtaRing[iEtaRing].Phi());
      hHCAL_D4_MExvsieta->Fill(iEtaRing - 41, vHOMET_EtaRing[iEtaRing].Px());
      hHCAL_D4_MEyvsieta->Fill(iEtaRing - 41, vHOMET_EtaRing[iEtaRing].Py());
      hHCAL_D4_METvsieta->Fill(iEtaRing - 41, vHOMET_EtaRing[iEtaRing].Pt());
      hHCAL_D4_SETvsieta->Fill(iEtaRing - 41, HOSET_EtaRing[iEtaRing]);
      hHCAL_D4_Occvsieta->Fill(iEtaRing - 41, HONActiveCells[iEtaRing]);
    }
  }

  //------------------------------------------------HF

  TLorentzVector vHFMET_EtaRing[83][2];
  int HFActiveRing[83][2];
  int HFNActiveCells[83][2];
  double HFSET_EtaRing[83][2];
  double HFMinEnergy_EtaRing[83][2];
  double HFMaxEnergy_EtaRing[83][2];

  for (int i = 0; i < 83; i++) {
    for (int j = 0; j < 2; j++) {
      HFActiveRing[i][j] = 0;
      HFNActiveCells[i][j] = 0;
      HFSET_EtaRing[i][j] = 0;
      HFMinEnergy_EtaRing[i][j] = 14E3;
      HFMaxEnergy_EtaRing[i][j] = -999;
    }
  }

  // Loop over HFRecHit's
  HFRecHitCollection::const_iterator hfrechit;
  int nHFrechit = 0;

  for (hfrechit = HFRecHits->begin(); hfrechit != HFRecHits->end(); hfrechit++) {
    nHFrechit++;

    HcalDetId det = hfrechit->id();
    double Energy = hfrechit->energy();
    Int_t depth = det.depth();
    Int_t ieta = det.ieta();
    Int_t iphi = det.iphi();
    int EtaRing = 41 + ieta;  // this counts from 0
    double eta = hHCAL_ieta_iphi_etaMap->getBinContent(EtaRing + 1, iphi);
    double phi = hHCAL_ieta_iphi_phiMap->getBinContent(EtaRing + 1, iphi);
    double theta = 2 * TMath::ATan(exp(-1 * eta));
    double ET = Energy * TMath::Sin(theta);

    TLorentzVector v_;
    if (Energy > 0)  // zero suppress
    {
      HFActiveRing[EtaRing][depth - 1] = 1;
      HFNActiveCells[EtaRing][depth - 1]++;
      HFSET_EtaRing[EtaRing][depth - 1] += ET;
      v_.SetPtEtaPhiE(ET, 0, phi, ET);
      vHFMET_EtaRing[EtaRing][depth - 1] -= v_;

      switch (depth) {
        case 1:
          hHCAL_D1_Occ_ieta_iphi->Fill(ieta, iphi);
          break;
        case 2:
          hHCAL_D2_Occ_ieta_iphi->Fill(ieta, iphi);
          break;
      }
    }

    if (Energy > HFMaxEnergy_EtaRing[EtaRing][depth - 1])
      HFMaxEnergy_EtaRing[EtaRing][depth - 1] = Energy;
    if (Energy < HFMinEnergy_EtaRing[EtaRing][depth - 1])
      HFMinEnergy_EtaRing[EtaRing][depth - 1] = Energy;

    switch (depth) {
      case 1:
        hHCAL_D1_energy_ieta_iphi->Fill(ieta, iphi, Energy);

        hHCAL_D1_energyvsieta->Fill(ieta, Energy);
        if (Energy > hHCAL_D1_Maxenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D1_Maxenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        if (Energy < hHCAL_D1_Minenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D1_Minenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        break;
      case 2:
        hHCAL_D2_energy_ieta_iphi->Fill(ieta, iphi, Energy);

        hHCAL_D2_energyvsieta->Fill(ieta, Energy);
        if (Energy > hHCAL_D2_Maxenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D2_Maxenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        if (Energy < hHCAL_D2_Minenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
          hHCAL_D2_Minenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
        break;
    }

  }  // end loop over HFRecHit's

  // Fill eta-ring MET quantities
  for (int iEtaRing = 0; iEtaRing < 83; iEtaRing++) {
    for (int jDepth = 0; jDepth < 2; jDepth++) {
      switch (jDepth + 1) {
        case 1:
          hHCAL_D1_Maxenergyvsieta->Fill(iEtaRing - 41, HFMaxEnergy_EtaRing[iEtaRing][jDepth]);
          hHCAL_D1_Minenergyvsieta->Fill(iEtaRing - 41, HFMinEnergy_EtaRing[iEtaRing][jDepth]);
          break;
        case 2:
          hHCAL_D2_Maxenergyvsieta->Fill(iEtaRing - 41, HFMaxEnergy_EtaRing[iEtaRing][jDepth]);
          hHCAL_D2_Minenergyvsieta->Fill(iEtaRing - 41, HFMinEnergy_EtaRing[iEtaRing][jDepth]);
          break;
      }

      if (HFActiveRing[iEtaRing][jDepth]) {
        switch (jDepth + 1) {
          case 1:

            hHCAL_D1_METPhivsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Phi());
            hHCAL_D1_MExvsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Px());
            hHCAL_D1_MEyvsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Py());
            hHCAL_D1_METvsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Pt());
            hHCAL_D1_SETvsieta->Fill(iEtaRing - 41, HFSET_EtaRing[iEtaRing][jDepth]);
            hHCAL_D1_Occvsieta->Fill(iEtaRing - 41, HFNActiveCells[iEtaRing][jDepth]);
            break;

          case 2:

            hHCAL_D2_METPhivsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Phi());
            hHCAL_D2_MExvsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Px());
            hHCAL_D2_MEyvsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Py());
            hHCAL_D2_METvsieta->Fill(iEtaRing - 41, vHFMET_EtaRing[iEtaRing][jDepth].Pt());
            hHCAL_D2_SETvsieta->Fill(iEtaRing - 41, HFSET_EtaRing[iEtaRing][jDepth]);
            hHCAL_D2_Occvsieta->Fill(iEtaRing - 41, HFNActiveCells[iEtaRing][jDepth]);
            break;
        }
      }
    }
  }
}
