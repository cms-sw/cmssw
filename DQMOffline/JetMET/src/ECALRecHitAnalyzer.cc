#include "DQMOffline/JetMET/interface/ECALRecHitAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

// author: Bobby Scurlock, University of Florida
// first version 11/20/2006

#define DEBUG(X)                   \
  {                                \
    if (debug_) {                  \
      std::cout << X << std::endl; \
    }                              \
  }

ECALRecHitAnalyzer::ECALRecHitAnalyzer(const edm::ParameterSet& iConfig) {
  // Retrieve Information from the Configuration File
  EBRecHitsLabel_ = consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitsLabel"));
  EERecHitsLabel_ = consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitsLabel"));
  caloGeomToken_ = esConsumes<edm::Transition::BeginRun>();
  FolderName_ = iConfig.getUntrackedParameter<std::string>("FolderName");
  debug_ = iConfig.getParameter<bool>("Debug");
  //  EBRecHitsLabel_= consumes<EcalRecHitCollection>(edm::InputTag(EBRecHitsLabel_));
  //  EERecHitsLabel_= consumes<EcalRecHitCollection>(edm::InputTag(EERecHitsLabel_));
}

void ECALRecHitAnalyzer::dqmbeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  CurrentEvent = -1;
  caloGeom_ = &iSetup.getData(caloGeomToken_);
  // Fill the geometry histograms
  FillGeometry(iSetup);
}

void ECALRecHitAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {
  // get ahold of back-end interface
  // Book Geometry Histograms
  ibooker.setCurrentFolder(FolderName_ + "/geometry");

  // ECAL barrel
  hEB_ieta_iphi_etaMap = ibooker.book2D("hEB_ieta_iphi_etaMap", "", 171, -85, 86, 360, 1, 361);
  hEB_ieta_iphi_phiMap = ibooker.book2D("hEB_ieta_iphi_phiMap", "", 171, -85, 86, 360, 1, 361);
  hEB_ieta_detaMap = ibooker.book1D("hEB_ieta_detaMap", "", 171, -85, 86);
  hEB_ieta_dphiMap = ibooker.book1D("hEB_ieta_dphiMap", "", 171, -85, 86);
  // ECAL +endcap
  hEEpZ_ix_iy_irMap = ibooker.book2D("hEEpZ_ix_iy_irMap", "", 100, 1, 101, 100, 1, 101);
  hEEpZ_ix_iy_xMap = ibooker.book2D("hEEpZ_ix_iy_xMap", "", 100, 1, 101, 100, 1, 101);
  hEEpZ_ix_iy_yMap = ibooker.book2D("hEEpZ_ix_iy_yMap", "", 100, 1, 101, 100, 1, 101);
  hEEpZ_ix_iy_zMap = ibooker.book2D("hEEpZ_ix_iy_zMap", "", 100, 1, 101, 100, 1, 101);
  hEEpZ_ix_iy_dxMap = ibooker.book2D("hEEpZ_ix_iy_dxMap", "", 100, 1, 101, 100, 1, 101);
  hEEpZ_ix_iy_dyMap = ibooker.book2D("hEEpZ_ix_iy_dyMap", "", 100, 1, 101, 100, 1, 101);
  // ECAL -endcap
  hEEmZ_ix_iy_irMap = ibooker.book2D("hEEmZ_ix_iy_irMap", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_ix_iy_xMap = ibooker.book2D("hEEmZ_ix_iy_xMap", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_ix_iy_yMap = ibooker.book2D("hEEmZ_ix_iy_yMap", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_ix_iy_zMap = ibooker.book2D("hEEmZ_ix_iy_zMap", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_ix_iy_dxMap = ibooker.book2D("hEEmZ_ix_iy_dxMap", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_ix_iy_dyMap = ibooker.book2D("hEEmZ_ix_iy_dyMap", "", 100, 1, 101, 100, 1, 101);

  // Initialize bins for geometry to -999 because z = 0 is a valid entry
  for (int i = 1; i <= 100; i++)
    for (int j = 1; j <= 100; j++) {
      hEEpZ_ix_iy_irMap->setBinContent(i, j, -999);
      hEEpZ_ix_iy_xMap->setBinContent(i, j, -999);
      hEEpZ_ix_iy_yMap->setBinContent(i, j, -999);
      hEEpZ_ix_iy_zMap->setBinContent(i, j, -999);
      hEEpZ_ix_iy_dxMap->setBinContent(i, j, -999);
      hEEpZ_ix_iy_dyMap->setBinContent(i, j, -999);

      hEEmZ_ix_iy_irMap->setBinContent(i, j, -999);
      hEEmZ_ix_iy_xMap->setBinContent(i, j, -999);
      hEEmZ_ix_iy_yMap->setBinContent(i, j, -999);
      hEEmZ_ix_iy_zMap->setBinContent(i, j, -999);
      hEEmZ_ix_iy_dxMap->setBinContent(i, j, -999);
      hEEmZ_ix_iy_dyMap->setBinContent(i, j, -999);
    }

  for (int i = 1; i <= 171; i++) {
    hEB_ieta_detaMap->setBinContent(i, -999);
    hEB_ieta_dphiMap->setBinContent(i, -999);
    for (int j = 1; j <= 360; j++) {
      hEB_ieta_iphi_etaMap->setBinContent(i, j, -999);
      hEB_ieta_iphi_phiMap->setBinContent(i, j, -999);
    }
  }

  // Book Data Histograms
  ibooker.setCurrentFolder(FolderName_);

  hECAL_Nevents = ibooker.book1D("hECAL_Nevents", "", 1, 0, 1);

  // Energy Histograms by logical index
  hEEpZ_energy_ix_iy = ibooker.book2D("hEEpZ_energy_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_energy_ix_iy = ibooker.book2D("hEEmZ_energy_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEB_energy_ieta_iphi = ibooker.book2D("hEB_energy_ieta_iphi", "", 171, -85, 86, 360, 1, 361);

  hEEpZ_Minenergy_ix_iy = ibooker.book2D("hEEpZ_Minenergy_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_Minenergy_ix_iy = ibooker.book2D("hEEmZ_Minenergy_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEB_Minenergy_ieta_iphi = ibooker.book2D("hEB_Minenergy_ieta_iphi", "", 171, -85, 86, 360, 1, 361);

  hEEpZ_Maxenergy_ix_iy = ibooker.book2D("hEEpZ_Maxenergy_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_Maxenergy_ix_iy = ibooker.book2D("hEEmZ_Maxenergy_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEB_Maxenergy_ieta_iphi = ibooker.book2D("hEB_Maxenergy_ieta_iphi", "", 171, -85, 86, 360, 1, 361);

  // need to initialize those
  for (int i = 1; i <= 171; i++)
    for (int j = 1; j <= 360; j++) {
      hEB_Maxenergy_ieta_iphi->setBinContent(i, j, -999);
      hEB_Minenergy_ieta_iphi->setBinContent(i, j, 14000);
    }
  for (int i = 1; i <= 100; i++)
    for (int j = 1; j <= 100; j++) {
      hEEpZ_Maxenergy_ix_iy->setBinContent(i, j, -999);
      hEEpZ_Minenergy_ix_iy->setBinContent(i, j, 14000);
      hEEmZ_Maxenergy_ix_iy->setBinContent(i, j, -999);
      hEEmZ_Minenergy_ix_iy->setBinContent(i, j, 14000);
    }

  // Occupancy Histograms by logical index
  hEEpZ_Occ_ix_iy = ibooker.book2D("hEEpZ_Occ_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEEmZ_Occ_ix_iy = ibooker.book2D("hEEmZ_Occ_ix_iy", "", 100, 1, 101, 100, 1, 101);
  hEB_Occ_ieta_iphi = ibooker.book2D("hEB_Occ_ieta_iphi", "", 171, -85, 86, 360, 1, 361);

  // Integrated Histograms
  if (finebinning_) {
    hEEpZ_energyvsir = ibooker.book2D("hEEpZ_energyvsir", "", 100, 1, 101, 20110, -10, 201);
    hEEmZ_energyvsir = ibooker.book2D("hEEmZ_energyvsir", "", 100, 1, 101, 20110, -10, 201);
    hEB_energyvsieta = ibooker.book2D("hEB_energyvsieta", "", 171, -85, 86, 20110, -10, 201);

    hEEpZ_Maxenergyvsir = ibooker.book2D("hEEpZ_Maxenergyvsir", "", 100, 1, 101, 20110, -10, 201);
    hEEmZ_Maxenergyvsir = ibooker.book2D("hEEmZ_Maxenergyvsir", "", 100, 1, 101, 20110, -10, 201);
    hEB_Maxenergyvsieta = ibooker.book2D("hEB_Maxenergyvsieta", "", 171, -85, 86, 20110, -10, 201);

    hEEpZ_Minenergyvsir = ibooker.book2D("hEEpZ_Minenergyvsir", "", 100, 1, 101, 20110, -10, 201);
    hEEmZ_Minenergyvsir = ibooker.book2D("hEEmZ_Minenergyvsir", "", 100, 1, 101, 20110, -10, 201);
    hEB_Minenergyvsieta = ibooker.book2D("hEB_Minenergyvsieta", "", 171, -85, 86, 20110, -10, 201);

    hEEpZ_SETvsir = ibooker.book2D("hEEpZ_SETvsir", "", 50, 1, 51, 20010, 0, 201);
    hEEmZ_SETvsir = ibooker.book2D("hEEmZ_SETvsir", "", 50, 1, 51, 20010, 0, 201);
    hEB_SETvsieta = ibooker.book2D("hEB_SETvsieta", "", 171, -85, 86, 20010, 0, 201);

    hEEpZ_METvsir = ibooker.book2D("hEEpZ_METvsir", "", 50, 1, 51, 20010, 0, 201);
    hEEmZ_METvsir = ibooker.book2D("hEEmZ_METvsir", "", 50, 1, 51, 20010, 0, 201);
    hEB_METvsieta = ibooker.book2D("hEB_METvsieta", "", 171, -85, 86, 20010, 0, 201);

    hEEpZ_METPhivsir = ibooker.book2D("hEEpZ_METPhivsir", "", 50, 1, 51, 80, -4, 4);
    hEEmZ_METPhivsir = ibooker.book2D("hEEmZ_METPhivsir", "", 50, 1, 51, 80, -4, 4);
    hEB_METPhivsieta = ibooker.book2D("hEB_METPhivsieta", "", 171, -85, 86, 80, -4, 4);

    hEEpZ_MExvsir = ibooker.book2D("hEEpZ_MExvsir", "", 50, 1, 51, 10010, -50, 51);
    hEEmZ_MExvsir = ibooker.book2D("hEEmZ_MExvsir", "", 50, 1, 51, 10010, -50, 51);
    hEB_MExvsieta = ibooker.book2D("hEB_MExvsieta", "", 171, -85, 86, 10010, -50, 51);

    hEEpZ_MEyvsir = ibooker.book2D("hEEpZ_MEyvsir", "", 50, 1, 51, 10010, -50, 51);
    hEEmZ_MEyvsir = ibooker.book2D("hEEmZ_MEyvsir", "", 50, 1, 51, 10010, -50, 51);
    hEB_MEyvsieta = ibooker.book2D("hEB_MEyvsieta", "", 171, -85, 86, 10010, -50, 51);

    hEEpZ_Occvsir = ibooker.book2D("hEEpZ_Occvsir", "", 50, 1, 51, 1000, 0, 1000);
    hEEmZ_Occvsir = ibooker.book2D("hEEmZ_Occvsir", "", 50, 1, 51, 1000, 0, 1000);
    hEB_Occvsieta = ibooker.book2D("hEB_Occvsieta", "", 171, -85, 86, 400, 0, 400);
  } else {
    hEEpZ_energyvsir = ibooker.book2D("hEEpZ_energyvsir", "", 100, 1, 101, 510, -10, 100);
    hEEmZ_energyvsir = ibooker.book2D("hEEmZ_energyvsir", "", 100, 1, 101, 510, -10, 100);
    hEB_energyvsieta = ibooker.book2D("hEB_energyvsieta", "", 171, -85, 86, 510, -10, 100);

    hEEpZ_Maxenergyvsir = ibooker.book2D("hEEpZ_Maxenergyvsir", "", 100, 1, 101, 510, -10, 100);
    hEEmZ_Maxenergyvsir = ibooker.book2D("hEEmZ_Maxenergyvsir", "", 100, 1, 101, 510, -10, 100);
    hEB_Maxenergyvsieta = ibooker.book2D("hEB_Maxenergyvsieta", "", 171, -85, 86, 510, -10, 100);

    hEEpZ_Minenergyvsir = ibooker.book2D("hEEpZ_Minenergyvsir", "", 100, 1, 101, 510, -10, 100);
    hEEmZ_Minenergyvsir = ibooker.book2D("hEEmZ_Minenergyvsir", "", 100, 1, 101, 510, -10, 100);
    hEB_Minenergyvsieta = ibooker.book2D("hEB_Minenergyvsieta", "", 171, -85, 86, 510, -10, 100);

    hEEpZ_SETvsir = ibooker.book2D("hEEpZ_SETvsir", "", 50, 1, 51, 510, 0, 100);
    hEEmZ_SETvsir = ibooker.book2D("hEEmZ_SETvsir", "", 50, 1, 51, 510, 0, 100);
    hEB_SETvsieta = ibooker.book2D("hEB_SETvsieta", "", 171, -85, 86, 510, 0, 100);

    hEEpZ_METvsir = ibooker.book2D("hEEpZ_METvsir", "", 50, 1, 51, 510, 0, 100);
    hEEmZ_METvsir = ibooker.book2D("hEEmZ_METvsir", "", 50, 1, 51, 510, 0, 100);
    hEB_METvsieta = ibooker.book2D("hEB_METvsieta", "", 171, -85, 86, 510, 0, 100);

    hEEpZ_METPhivsir = ibooker.book2D("hEEpZ_METPhivsir", "", 50, 1, 51, 80, -4, 4);
    hEEmZ_METPhivsir = ibooker.book2D("hEEmZ_METPhivsir", "", 50, 1, 51, 80, -4, 4);
    hEB_METPhivsieta = ibooker.book2D("hEB_METPhivsieta", "", 171, -85, 86, 80, -4, 4);

    hEEpZ_MExvsir = ibooker.book2D("hEEpZ_MExvsir", "", 50, 1, 51, 510, -50, 51);
    hEEmZ_MExvsir = ibooker.book2D("hEEmZ_MExvsir", "", 50, 1, 51, 510, -50, 51);
    hEB_MExvsieta = ibooker.book2D("hEB_MExvsieta", "", 171, -85, 86, 510, -50, 51);

    hEEpZ_MEyvsir = ibooker.book2D("hEEpZ_MEyvsir", "", 50, 1, 51, 510, -50, 51);
    hEEmZ_MEyvsir = ibooker.book2D("hEEmZ_MEyvsir", "", 50, 1, 51, 510, -50, 51);
    hEB_MEyvsieta = ibooker.book2D("hEB_MEyvsieta", "", 171, -85, 86, 510, -50, 51);

    hEEpZ_Occvsir = ibooker.book2D("hEEpZ_Occvsir", "", 50, 1, 51, 1000, 0, 1000);
    hEEmZ_Occvsir = ibooker.book2D("hEEmZ_Occvsir", "", 50, 1, 51, 1000, 0, 1000);
    hEB_Occvsieta = ibooker.book2D("hEB_Occvsieta", "", 171, -85, 86, 400, 0, 400);
  }
}

void ECALRecHitAnalyzer::FillGeometry(const edm::EventSetup& iSetup) {
  // Fill geometry histograms
  using namespace edm;
  //const auto& cG = iSetup.getData(caloGeomToken_);
  //----Fill Ecal Barrel----//
  const CaloSubdetectorGeometry* EBgeom = caloGeom_->getSubdetectorGeometry(DetId::Ecal, 1);
  int n = 0;
  std::vector<DetId> EBids = EBgeom->getValidDetIds(DetId::Ecal, 1);
  for (std::vector<DetId>::iterator i = EBids.begin(); i != EBids.end(); i++) {
    n++;
    auto cell = EBgeom->getGeometry(*i);
    //GlobalPoint p = cell->getPosition();

    EBDetId EcalID(i->rawId());

    int Crystal_ieta = EcalID.ieta();
    int Crystal_iphi = EcalID.iphi();
    double Crystal_eta = cell->getPosition().eta();
    double Crystal_phi = cell->getPosition().phi();
    hEB_ieta_iphi_etaMap->setBinContent(Crystal_ieta + 86, Crystal_iphi, Crystal_eta);
    hEB_ieta_iphi_phiMap->setBinContent(Crystal_ieta + 86, Crystal_iphi, (Crystal_phi * 180 / M_PI));

    DEBUG(" Crystal " << n);
    DEBUG("  ieta, iphi = " << Crystal_ieta << ", " << Crystal_iphi);
    DEBUG("   eta,  phi = " << cell->getPosition().eta() << ", " << cell->getPosition().phi());
    DEBUG(" ");
  }
  //----Fill Ecal Endcap----------//
  const CaloSubdetectorGeometry* EEgeom = caloGeom_->getSubdetectorGeometry(DetId::Ecal, 2);
  n = 0;
  std::vector<DetId> EEids = EEgeom->getValidDetIds(DetId::Ecal, 2);
  for (std::vector<DetId>::iterator i = EEids.begin(); i != EEids.end(); i++) {
    n++;
    auto cell = EEgeom->getGeometry(*i);
    //GlobalPoint p = cell->getPosition();
    EEDetId EcalID(i->rawId());
    int Crystal_zside = EcalID.zside();
    int Crystal_ix = EcalID.ix();
    int Crystal_iy = EcalID.iy();
    Float_t ix_ = Crystal_ix - 50.5;
    Float_t iy_ = Crystal_iy - 50.5;
    Int_t ir = (Int_t)sqrt(ix_ * ix_ + iy_ * iy_);

    //double Crystal_eta = cell->getPosition().eta();
    //double Crystal_phi = cell->getPosition().phi();
    double Crystal_x = cell->getPosition().x();
    double Crystal_y = cell->getPosition().y();
    double Crystal_z = cell->getPosition().z();
    // ECAL -endcap
    if (Crystal_zside == -1) {
      hEEmZ_ix_iy_irMap->setBinContent(Crystal_ix, Crystal_iy, ir);
      hEEmZ_ix_iy_xMap->setBinContent(Crystal_ix, Crystal_iy, Crystal_x);
      hEEmZ_ix_iy_yMap->setBinContent(Crystal_ix, Crystal_iy, Crystal_y);
      hEEmZ_ix_iy_zMap->setBinContent(Crystal_ix, Crystal_iy, Crystal_z);
    }
    // ECAL +endcap
    if (Crystal_zside == 1) {
      hEEpZ_ix_iy_irMap->setBinContent(Crystal_ix, Crystal_iy, ir);
      hEEpZ_ix_iy_xMap->setBinContent(Crystal_ix, Crystal_iy, Crystal_x);
      hEEpZ_ix_iy_yMap->setBinContent(Crystal_ix, Crystal_iy, Crystal_y);
      hEEpZ_ix_iy_zMap->setBinContent(Crystal_ix, Crystal_iy, Crystal_z);
    }

    DEBUG(" Crystal " << n);
    DEBUG("  side = " << Crystal_zside);
    DEBUG("   ix, iy = " << Crystal_ix << ", " << Crystal_iy);
    DEBUG("    x,  y = " << Crystal_x << ", " << Crystal_y);
    ;
    DEBUG(" ");
  }

  //-------Set the cell size for each (ieta, iphi) bin-------//
  //double currentHighEdge_eta = 0;
  for (int ieta = 1; ieta <= 85; ieta++) {
    int ieta_ = 86 + ieta;

    double eta = hEB_ieta_iphi_etaMap->getBinContent(ieta_, 1);
    double etam1 = -999;

    if (ieta == 1)
      etam1 = hEB_ieta_iphi_etaMap->getBinContent(85, 1);
    else
      etam1 = hEB_ieta_iphi_etaMap->getBinContent(ieta_ - 1, 1);

    //double phi = hEB_ieta_iphi_phiMap->getBinContent(ieta_, 1);
    double deta = fabs(eta - etam1);
    double dphi = fabs(hEB_ieta_iphi_phiMap->getBinContent(ieta_, 1) - hEB_ieta_iphi_phiMap->getBinContent(ieta_, 2));

    hEB_ieta_detaMap->setBinContent(ieta_, deta);      // positive rings
    hEB_ieta_dphiMap->setBinContent(ieta_, dphi);      // positive rings
    hEB_ieta_detaMap->setBinContent(86 - ieta, deta);  // negative rings
    hEB_ieta_dphiMap->setBinContent(86 - ieta, dphi);  // negative rings
  }
}

void ECALRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  CurrentEvent++;
  DEBUG("Event: " << CurrentEvent);
  WriteECALRecHits(iEvent, iSetup);
  hECAL_Nevents->Fill(0.5);
}

void ECALRecHitAnalyzer::WriteECALRecHits(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits;
  iEvent.getByToken(EBRecHitsLabel_, EBRecHits);
  iEvent.getByToken(EERecHitsLabel_, EERecHits);
  DEBUG("Got ECALRecHits");

  //const CaloSubdetectorGeometry* EBgeom=cG.getSubdetectorGeometry(DetId::Ecal,1);
  //const CaloSubdetectorGeometry* EEgeom=cG.getSubdetectorGeometry(DetId::Ecal,2);
  DEBUG("Got Geometry");

  TLorentzVector vEBMET_EtaRing[171];
  int EBActiveRing[171];
  int EBNActiveCells[171];
  double EBSET_EtaRing[171];
  double EBMaxEnergy_EtaRing[171];
  double EBMinEnergy_EtaRing[171];
  double EBenergy_EtaRing[171];

  for (int i = 0; i < 171; i++) {
    EBActiveRing[i] = 0;
    EBNActiveCells[i] = 0;
    EBSET_EtaRing[i] = 0.0;
    EBMaxEnergy_EtaRing[i] = -999;
    EBMinEnergy_EtaRing[i] = 14E3;
    EBenergy_EtaRing[i] = 0.0;
  }

  edm::LogInfo("OutputInfo") << "Looping over EB" << std::endl;

  EBRecHitCollection::const_iterator ebrechit;
  //int nEBrechit = 0;

  for (ebrechit = EBRecHits->begin(); ebrechit != EBRecHits->end(); ebrechit++) {
    EBDetId det = ebrechit->id();
    double Energy = ebrechit->energy();
    Int_t ieta = det.ieta();
    Int_t iphi = det.iphi();
    int EtaRing = 85 + ieta;  // this counts from 0
    double eta = hEB_ieta_iphi_etaMap->getBinContent(EtaRing + 1, iphi);
    double phi = hEB_ieta_iphi_phiMap->getBinContent(EtaRing + 1, iphi);
    double theta = 2 * TMath::ATan(exp(-1 * eta));
    double ET = Energy * TMath::Sin(theta);
    TLorentzVector v_;

    if (Energy > EBMaxEnergy_EtaRing[EtaRing])
      EBMaxEnergy_EtaRing[EtaRing] = Energy;
    if (Energy < EBMinEnergy_EtaRing[EtaRing])
      EBMinEnergy_EtaRing[EtaRing] = Energy;

    if (Energy > 0) {
      EBActiveRing[EtaRing] = 1;
      EBNActiveCells[EtaRing]++;
      EBSET_EtaRing[EtaRing] += ET;
      v_.SetPtEtaPhiE(ET, 0, phi, ET);
      vEBMET_EtaRing[EtaRing] -= v_;
      EBenergy_EtaRing[EtaRing] += Energy;
      hEB_Occ_ieta_iphi->Fill(ieta, iphi);
    }

    hEB_energy_ieta_iphi->Fill(ieta, iphi, Energy);
    if (Energy > hEB_Maxenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
      hEB_Maxenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);
    if (Energy < hEB_Minenergy_ieta_iphi->getBinContent(EtaRing + 1, iphi))
      hEB_Minenergy_ieta_iphi->setBinContent(EtaRing + 1, iphi, Energy);

  }  // loop over EB

  for (int iEtaRing = 0; iEtaRing < 171; iEtaRing++) {
    hEB_Minenergyvsieta->Fill(iEtaRing - 85, EBMinEnergy_EtaRing[iEtaRing]);
    hEB_Maxenergyvsieta->Fill(iEtaRing - 85, EBMaxEnergy_EtaRing[iEtaRing]);

    if (EBActiveRing[iEtaRing]) {
      hEB_METvsieta->Fill(iEtaRing - 85, vEBMET_EtaRing[iEtaRing].Pt());
      hEB_METPhivsieta->Fill(iEtaRing - 85, vEBMET_EtaRing[iEtaRing].Phi());
      hEB_MExvsieta->Fill(iEtaRing - 85, vEBMET_EtaRing[iEtaRing].Px());
      hEB_MEyvsieta->Fill(iEtaRing - 85, vEBMET_EtaRing[iEtaRing].Py());
      hEB_SETvsieta->Fill(iEtaRing - 85, EBSET_EtaRing[iEtaRing]);
      hEB_Occvsieta->Fill(iEtaRing - 85, EBNActiveCells[iEtaRing]);
      hEB_energyvsieta->Fill(iEtaRing - 85, EBenergy_EtaRing[iEtaRing]);
    }
  }

  TLorentzVector vEEpZMET_EtaRing[101];
  int EEpZActiveRing[101];
  int EEpZNActiveCells[101];
  double EEpZSET_EtaRing[101];
  double EEpZMaxEnergy_EtaRing[101];
  double EEpZMinEnergy_EtaRing[101];

  TLorentzVector vEEmZMET_EtaRing[101];
  int EEmZActiveRing[101];
  int EEmZNActiveCells[101];
  double EEmZSET_EtaRing[101];
  double EEmZMaxEnergy_EtaRing[101];
  double EEmZMinEnergy_EtaRing[101];

  for (int i = 0; i < 101; i++) {
    EEpZActiveRing[i] = 0;
    EEpZNActiveCells[i] = 0;
    EEpZSET_EtaRing[i] = 0.0;
    EEpZMaxEnergy_EtaRing[i] = -999;
    EEpZMinEnergy_EtaRing[i] = 14E3;

    EEmZActiveRing[i] = 0;
    EEmZNActiveCells[i] = 0;
    EEmZSET_EtaRing[i] = 0.0;
    EEmZMaxEnergy_EtaRing[i] = -999;
    EEmZMinEnergy_EtaRing[i] = 14E3;
  }

  edm::LogInfo("OutputInfo") << "Looping over EE" << std::endl;
  EERecHitCollection::const_iterator eerechit;
  //int nEErechit = 0;
  for (eerechit = EERecHits->begin(); eerechit != EERecHits->end(); eerechit++) {
    EEDetId det = eerechit->id();
    double Energy = eerechit->energy();
    Int_t ix = det.ix();
    Int_t iy = det.iy();
    //Float_t ix_ = (Float_t)-999;
    //Float_t iy_ = (Float_t)-999;
    Int_t ir = -999;
    //    edm::LogInfo("OutputInfo") << ix << " " << iy << " " << ix_ << " " << iy_ << " " << ir << std::endl;

    double x = -999;
    double y = -999;
    double z = -999;
    double theta = -999;
    double phi = -999;

    int Crystal_zside = det.zside();

    if (Crystal_zside == -1) {
      ir = (Int_t)hEEmZ_ix_iy_irMap->getBinContent(ix, iy);
      x = hEEmZ_ix_iy_xMap->getBinContent(ix, iy);
      y = hEEmZ_ix_iy_yMap->getBinContent(ix, iy);
      z = hEEmZ_ix_iy_zMap->getBinContent(ix, iy);
    }
    if (Crystal_zside == 1) {
      ir = (Int_t)hEEpZ_ix_iy_irMap->getBinContent(ix, iy);
      x = hEEpZ_ix_iy_xMap->getBinContent(ix, iy);
      y = hEEpZ_ix_iy_yMap->getBinContent(ix, iy);
      z = hEEpZ_ix_iy_zMap->getBinContent(ix, iy);
    }
    TVector3 pos_vector(x, y, z);
    phi = pos_vector.Phi();
    theta = pos_vector.Theta();
    double ET = Energy * TMath::Sin(theta);
    TLorentzVector v_;

    if (Crystal_zside == -1) {
      if (Energy > 0) {
        EEmZActiveRing[ir] = 1;
        EEmZNActiveCells[ir]++;
        EEmZSET_EtaRing[ir] += ET;
        v_.SetPtEtaPhiE(ET, 0, phi, ET);
        vEEmZMET_EtaRing[ir] -= v_;
        hEEmZ_Occ_ix_iy->Fill(ix, iy);
      }
      hEEmZ_energyvsir->Fill(ir, Energy);
      hEEmZ_energy_ix_iy->Fill(ix, iy, Energy);

      if (Energy > EEmZMaxEnergy_EtaRing[ir])
        EEmZMaxEnergy_EtaRing[ir] = Energy;
      if (Energy < EEmZMinEnergy_EtaRing[ir])
        EEmZMinEnergy_EtaRing[ir] = Energy;

      if (Energy > hEEmZ_Maxenergy_ix_iy->getBinContent(ix, iy))
        hEEmZ_Maxenergy_ix_iy->setBinContent(ix, iy, Energy);
      if (Energy < hEEmZ_Minenergy_ix_iy->getBinContent(ix, iy))
        hEEmZ_Minenergy_ix_iy->setBinContent(ix, iy, Energy);
    }
    if (Crystal_zside == 1) {
      if (Energy > 0) {
        EEpZActiveRing[ir] = 1;
        EEpZNActiveCells[ir]++;
        EEpZSET_EtaRing[ir] += ET;
        v_.SetPtEtaPhiE(ET, 0, phi, ET);
        vEEpZMET_EtaRing[ir] -= v_;
        hEEpZ_Occ_ix_iy->Fill(ix, iy);
      }
      hEEpZ_energyvsir->Fill(ir, Energy);
      hEEpZ_energy_ix_iy->Fill(ix, iy, Energy);

      if (Energy > EEpZMaxEnergy_EtaRing[ir])
        EEpZMaxEnergy_EtaRing[ir] = Energy;
      if (Energy < EEpZMinEnergy_EtaRing[ir])
        EEpZMinEnergy_EtaRing[ir] = Energy;
      if (Energy > hEEpZ_Maxenergy_ix_iy->getBinContent(ix, iy))
        hEEpZ_Maxenergy_ix_iy->setBinContent(ix, iy, Energy);
      if (Energy < hEEpZ_Minenergy_ix_iy->getBinContent(ix, iy))
        hEEpZ_Minenergy_ix_iy->setBinContent(ix, iy, Energy);
    }
  }  // loop over EE
  edm::LogInfo("OutputInfo") << "Done Looping over EE" << std::endl;
  for (int iEtaRing = 0; iEtaRing < 101; iEtaRing++) {
    hEEpZ_Maxenergyvsir->Fill(iEtaRing, EEpZMaxEnergy_EtaRing[iEtaRing]);
    hEEpZ_Minenergyvsir->Fill(iEtaRing, EEpZMinEnergy_EtaRing[iEtaRing]);
    hEEmZ_Maxenergyvsir->Fill(iEtaRing, EEmZMaxEnergy_EtaRing[iEtaRing]);
    hEEmZ_Minenergyvsir->Fill(iEtaRing, EEmZMinEnergy_EtaRing[iEtaRing]);

    if (EEpZActiveRing[iEtaRing]) {
      hEEpZ_METvsir->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Pt());
      hEEpZ_METPhivsir->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Phi());
      hEEpZ_MExvsir->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Px());
      hEEpZ_MEyvsir->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Py());
      hEEpZ_SETvsir->Fill(iEtaRing, EEpZSET_EtaRing[iEtaRing]);
      hEEpZ_Occvsir->Fill(iEtaRing, EEpZNActiveCells[iEtaRing]);
    }

    if (EEmZActiveRing[iEtaRing]) {
      hEEmZ_METvsir->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Pt());
      hEEmZ_METPhivsir->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Phi());
      hEEmZ_MExvsir->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Px());
      hEEmZ_MEyvsir->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Py());
      hEEmZ_SETvsir->Fill(iEtaRing, EEmZSET_EtaRing[iEtaRing]);
      hEEmZ_Occvsir->Fill(iEtaRing, EEmZNActiveCells[iEtaRing]);
    }
  }
  edm::LogInfo("OutputInfo") << "Done ..." << std::endl;
}  // loop over RecHits
