#include "DQM/L1TMonitor/interface/L1TCaloLayer1Summary.h"

L1TCaloLayer1Summary::L1TCaloLayer1Summary(const edm::ParameterSet& iConfig)
    : caloLayer1CICADAScoreToken_(
          consumes<l1t::CICADABxCollection>(iConfig.getParameter<edm::InputTag>("caloLayer1CICADAScore"))),
      gtCICADAScoreToken_(consumes<l1t::CICADABxCollection>(iConfig.getParameter<edm::InputTag>("gtCICADAScore"))),
      simCICADAScoreToken_(consumes<l1t::CICADABxCollection>(iConfig.getParameter<edm::InputTag>("simCICADAScore"))),
      caloLayer1RegionsToken_(
          consumes<L1CaloRegionCollection>(iConfig.getParameter<edm::InputTag>("caloLayer1Regions"))),
      simRegionsToken_(consumes<L1CaloRegionCollection>(iConfig.getParameter<edm::InputTag>("simRegions"))),
      fedRawData_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("fedRawDataLabel"))),
      histFolder_(iConfig.getParameter<std::string>("histFolder")) {}

// ------------ method called for each event  ------------
void L1TCaloLayer1Summary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<FEDRawDataCollection> fedRawDataCollection;
  iEvent.getByToken(fedRawData_, fedRawDataCollection);
  if (fedRawDataCollection.isValid()) {
    for (int iFed = 1354; iFed < 1360; iFed += 2) {
      const FEDRawData& fedRawData = fedRawDataCollection->FEDData(iFed);
      if (fedRawData.size() == 0) {
        continue;
      }
      const uint64_t* fedRawDataArray = (const uint64_t*)fedRawData.data();
      UCTDAQRawData daqData(fedRawDataArray);

      if (daqData.nAMCs() == 7) {
        UCTAMCRawData amcSlot7(daqData.amcPayload(3));
        if (amcSlot7.amcNo() != 7) {
          std::cout << "Wrong AMC No: " << amcSlot7.amcNo() << std::endl;
        } else {
          histoSlot7MinusDaqBxid->Fill(amcSlot7.BXID() - daqData.BXID());
        }
      }
    }
  }

  L1CaloRegionCollection caloLayer1Regions = iEvent.get(caloLayer1RegionsToken_);
  L1CaloRegionCollection simRegions = iEvent.get(simRegionsToken_);

  bool foundMatrix[2][18][18] = {};
  int etMatrix[2][18][18] = {};

  int nRegions = caloLayer1Regions.size();
  for (int iRegion = 0; iRegion < nRegions; iRegion++) {
    L1CaloRegion cRegion = caloLayer1Regions[iRegion];
    L1CaloRegion sRegion = simRegions[iRegion];

    foundMatrix[0][cRegion.gctEta()][cRegion.gctPhi()] = true;
    etMatrix[0][cRegion.gctEta()][cRegion.gctPhi()] = cRegion.et();
    foundMatrix[1][sRegion.gctEta()][sRegion.gctPhi()] = true;
    etMatrix[1][sRegion.gctEta()][sRegion.gctPhi()] = sRegion.et();
  }
  int iRegion = 0;
  for (int iEta = 0; iEta < 18; iEta++) {
    for (int iPhi = 0; iPhi < 18; iPhi++) {
      if (foundMatrix[0][iEta][iPhi] && foundMatrix[1][iEta][iPhi]) {
        histoCaloRegions->Fill(iRegion, etMatrix[0][iEta][iPhi]);
        histoSimRegions->Fill(iRegion, etMatrix[1][iEta][iPhi]);
        histoCaloMinusSimRegions->Fill(iRegion, etMatrix[0][iEta][iPhi] - etMatrix[1][iEta][iPhi]);
        iRegion++;
      }
    }
  }

  float caloCICADAScore = iEvent.get(caloLayer1CICADAScoreToken_)[0];
  auto gtCICADAScores = iEvent.get(gtCICADAScoreToken_);
  float simCICADAScore = iEvent.get(simCICADAScoreToken_)[0];

  histoSimCICADAScore->Fill(simCICADAScore);
  histoCaloMinusSim->Fill(caloCICADAScore - simCICADAScore);

  uint32_t bx0Idx;
  if (gtCICADAScores.size() == 30) {
    bx0Idx = 12;
  } else if (gtCICADAScores.size() == 5) {
    bx0Idx = 2;
  } else {
    bx0Idx = 2;
  }

  histoCaloLayer1CICADAScore->Fill(caloCICADAScore);
  histoGtCICADAScore->Fill(gtCICADAScores[bx0Idx]);
  histoCaloMinusGt->Fill(gtCICADAScores[bx0Idx] - caloCICADAScore);
}

void L1TCaloLayer1Summary::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(histFolder_);
  histoSlot7MinusDaqBxid = ibooker.book1D("slot7BXID", "Slot 7- DAQ BXID", 50, -20, 20);

  ibooker.setCurrentFolder(histFolder_ + "/CICADAScore");
  histoCaloLayer1CICADAScore = ibooker.book1D("caloLayer1CICADAScore", "CaloLayer1 CICADAScore", 50, 0, 200);
  histoGtCICADAScore = ibooker.book1D("gtCICADAScore", "GT CICADAScore at BX0", 50, 0, 200);
  histoCaloMinusGt = ibooker.book1D("caloMinusGtCICADAScore", "CaloLayer1 - GT CICADAScore at BX0", 50, -50, 50);
  histoSimCICADAScore = ibooker.book1D("simCaloLayer1CICADAScore", "simCaloLayer1 CICADAScore", 50, 0, 200);
  histoCaloMinusSim = ibooker.book1D("caloMinusSimCICADAScore", "CaloLayer1 - simCaloLayer1 CICADAScore", 50, -50, 50);

  ibooker.setCurrentFolder(histFolder_ + "/Regions");
  histoCaloMinusSimRegions = ibooker.book2D(
      "caloMinusSumRegions", "CaloLayer1 - simCaloLayer1 Regions;Region;ET", 252, -0.5, 252.5, 100, -400, 400);
  histoCaloRegions = ibooker.book2D("caloLayer1Regions", "CaloLayer1 Regions;Region;ET", 252, -0.5, 252.5, 100, 0, 800);
  histoSimRegions =
      ibooker.book2D("simCaloLayer1Regions", "simCaloLayer1 Regions;Region;ET", 252, -0.5, 252.5, 100, 0, 800);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TCaloLayer1Summary::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //edm::ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("ctfWithMaterialTracks"));
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloLayer1Summary);