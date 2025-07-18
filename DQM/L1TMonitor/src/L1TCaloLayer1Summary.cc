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
    for (int iFed = FEDNumbering::MINRCTFEDID + 4; iFed < FEDNumbering::MAXRCTFEDID; iFed += 2) {
      const FEDRawData& fedRawData = fedRawDataCollection->FEDData(iFed);
      if (fedRawData.size() == 0) {
        continue;
      }
      const uint64_t* fedRawDataArray = (const uint64_t*)fedRawData.data();
      UCTDAQRawData daqData(fedRawDataArray);

      if (daqData.nAMCs() == 7) {
        UCTAMCRawData amcSlot7(daqData.amcPayload(3));
        if (amcSlot7.amcNo() == 7) {
          histoSlot7MinusDaqBxid->Fill(amcSlot7.BXID() - daqData.BXID());
        }
      }
    }
  }

  L1CaloRegionCollection caloLayer1Regions = iEvent.get(caloLayer1RegionsToken_);
  L1CaloRegionCollection simRegions = iEvent.get(simRegionsToken_);
  int nRegions = caloLayer1Regions.size();

  unsigned int maxEtaIdx = 0;
  for (int iRegion = 0; iRegion < nRegions; iRegion++) {
    if (maxEtaIdx < caloLayer1Regions[iRegion].gctEta()) {
      maxEtaIdx = caloLayer1Regions[iRegion].gctEta();
    }
  }
  int matrixSize = maxEtaIdx + 1;

  bool foundMatrix[2][matrixSize][matrixSize];
  int etMatrix[2][matrixSize][matrixSize];
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < matrixSize; j++) {
      for (int k = 0; k < matrixSize; k++) {
        foundMatrix[i][j][k] = false;
        etMatrix[i][j][k] = 0;
      }
    }
  }

  for (int iRegion = 0; iRegion < nRegions; iRegion++) {
    L1CaloRegion cRegion = caloLayer1Regions[iRegion];
    L1CaloRegion sRegion = simRegions[iRegion];

    foundMatrix[0][cRegion.gctEta()][cRegion.gctPhi()] = true;
    etMatrix[0][cRegion.gctEta()][cRegion.gctPhi()] = cRegion.et();
    foundMatrix[1][sRegion.gctEta()][sRegion.gctPhi()] = true;
    etMatrix[1][sRegion.gctEta()][sRegion.gctPhi()] = sRegion.et();
  }
  int iRegion = 0;
  for (int iEta = 0; iEta < matrixSize; iEta++) {
    for (int iPhi = 0; iPhi < matrixSize; iPhi++) {
      if (foundMatrix[0][iEta][iPhi] && foundMatrix[1][iEta][iPhi]) {
        histoCaloRegions->Fill(iRegion, etMatrix[0][iEta][iPhi]);
        histoSimRegions->Fill(iRegion, etMatrix[1][iEta][iPhi]);
        histoCaloMinusSimRegions->Fill(iRegion, etMatrix[0][iEta][iPhi] - etMatrix[1][iEta][iPhi]);
        iRegion++;
      }
    }
  }

  auto caloCICADAScores = iEvent.get(caloLayer1CICADAScoreToken_);
  const auto& gtCICADAScores = iEvent.get(gtCICADAScoreToken_);
  auto simCICADAScores = iEvent.get(simCICADAScoreToken_);

  if (caloCICADAScores.size() > 0) {
    histoCaloLayer1CICADAScore->Fill(caloCICADAScores[0]);
    if (gtCICADAScores.size() > 0) {
      histoGtCICADAScore->Fill(gtCICADAScores.at(0, 0));
      histoCaloMinusGt->Fill(caloCICADAScores[0] - gtCICADAScores.at(0, 0));
    }
    if (simCICADAScores.size() > 0) {
      histoSimCICADAScore->Fill(simCICADAScores[0]);
      histoCaloMinusSim->Fill(caloCICADAScores[0] - simCICADAScores[0]);
    }
  }
}

void L1TCaloLayer1Summary::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(histFolder_);
  histoSlot7MinusDaqBxid = ibooker.book1D("slot7BXID", "Slot 7- DAQ BXID", 50, -20, 20);

  ibooker.setCurrentFolder(histFolder_ + "/CICADAScore");
  histoCaloLayer1CICADAScore = ibooker.book1D("caloLayer1CICADAScore", "CaloLayer1 CICADAScore", 50, 0, 200);
  histoGtCICADAScore = ibooker.book1D("gtCICADAScore", "GT CICADAScore at BX0", 50, 0, 200);
  histoCaloMinusGt = ibooker.book1D("caloMinusGtCICADAScore", "CaloLayer1 - GT CICADAScore at BX0", 50, -50, 50);
  histoSimCICADAScore =
      ibooker.book1D("simCaloLayer1CICADAScore", "simCaloLayer1 CICADAScore (input: DAQ regions)", 50, 0, 200);
  histoCaloMinusSim = ibooker.book1D(
      "caloMinusSimCICADAScore", "CaloLayer1 - simCaloLayer1 (input: DAQ regions) CICADAScore", 50, -50, 50);

  ibooker.setCurrentFolder(histFolder_ + "/Regions");
  histoCaloMinusSimRegions =
      ibooker.book2D("caloMinusSumRegions",
                     "CaloLayer1 - simCaloLayer1 (input: DAQ trigger primatives) Regions;Region;ET Difference",
                     252,
                     -0.5,
                     252.5,
                     100,
                     -400,
                     400);
  histoCaloRegions = ibooker.book2D("caloLayer1Regions", "CaloLayer1 Regions;Region;ET", 252, -0.5, 252.5, 100, 0, 800);
  histoSimRegions = ibooker.book2D("simCaloLayer1Regions",
                                   "simCaloLayer1 Regions (input: DAQ trigger primatives);Region;ET",
                                   252,
                                   -0.5,
                                   252.5,
                                   100,
                                   0,
                                   800);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TCaloLayer1Summary::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // l1tCaloLayer1Summary
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("caloLayer1CICADAScore", edm::InputTag("caloLayer1Digis", "CICADAScore"));
  desc.add<edm::InputTag>("gtCICADAScore", edm::InputTag("gtTestcrateStage2Digis", "CICADAScore"));
  desc.add<edm::InputTag>("simCICADAScore", edm::InputTag("simCaloStage2Layer1Summary", "CICADAScore"));
  desc.add<edm::InputTag>("caloLayer1Regions", edm::InputTag("caloLayer1Digis"));
  desc.add<edm::InputTag>("simRegions", edm::InputTag("simCaloStage2Layer1Digis"));
  desc.add<edm::InputTag>("fedRawDataLabel", edm::InputTag("rawDataCollector"));
  desc.add<std::string>("histFolder", "L1T/L1TCaloLayer1Summary");
  descriptions.add("l1tCaloLayer1Summary", desc);
}