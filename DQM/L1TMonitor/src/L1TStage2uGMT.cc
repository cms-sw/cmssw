#include "DQM/L1TMonitor/interface/L1TStage2uGMT.h"


L1TStage2uGMT::L1TStage2uGMT(const edm::ParameterSet& ps)
    : ugmtBMTFToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("bmtfSource"))),
      ugmtOMTFToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("omtfSource"))),
      ugmtEMTFToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      ugmtMuonToken(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2uGMT::~L1TStage2uGMT() {}

void L1TStage2uGMT::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2uGMT::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2uGMT::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // BMTF Input
  ibooker.setCurrentFolder(monitorDir + "/BMTFInput");

  ugmtBMTFBX = ibooker.book1D("ugmtBMTFBX", "uGMT BMTF Input BX", 5, -2.5, 2.5);
  ugmtBMTFBX->setAxisTitle("BX (Non-zero values prescaled by 107)", 1);

  ugmtBMTFhwPt = ibooker.book1D("ugmtBMTFhwPt", "uGMT BMTF Input p_{T}", 512, -0.5, 511.5);
  ugmtBMTFhwPt->setAxisTitle("Hardware p_{T}", 1);

  ugmtBMTFhwEta = ibooker.book1D("ugmtBMTFhwEta", "uGMT BMTF Input #eta", 201, -100.5, 100.5);
  ugmtBMTFhwEta->setAxisTitle("Hardware #eta", 1);
  
  ugmtBMTFhwPhi = ibooker.book1D("ugmtBMTFhwPhi", "uGMT BMTF Input #phi", 91, -15.5, 75.5);
  ugmtBMTFhwPhi->setAxisTitle("Hardware #phi", 1);

  ugmtBMTFglbPhi = ibooker.book1D("ugmtBMTFglbhwPhi", "uGMT BMTF Input #phi", 576, -0.5, 575.5);
  ugmtBMTFglbPhi->setAxisTitle("Global Hardware #phi", 1);

  ugmtBMTFhwSign = ibooker.book1D("ugmtBMTFhwSign", "uGMT BMTF Input Sign", 2, -0.5, 1.5);
  ugmtBMTFhwSign->setAxisTitle("Hardware Sign", 1);

  ugmtBMTFhwSignValid = ibooker.book1D("ugmtBMTFhwSignValid", "uGMT BMTF Input SignValid", 2, -0.5, 1.5);
  ugmtBMTFhwSignValid->setAxisTitle("SignValid", 1);

  ugmtBMTFhwQual = ibooker.book1D("ugmtBMTFhwQual", "uGMT BMTF Input Quality", 16, -0.5, 15.5);
  ugmtBMTFhwQual->setAxisTitle("Quality", 1);

  ugmtBMTFlink = ibooker.book1D("ugmtBMTFlink", "uGMT BMTF Input Link", 12, 47.5, 59.5);
  ugmtBMTFlink->setAxisTitle("Link", 1);

  // OMTF Input
  ibooker.setCurrentFolder(monitorDir + "/OMTFInput");

  ugmtOMTFBX = ibooker.book1D("ugmtOMTFBX", "uGMT OMTF Input BX", 5, -2.5, 2.5);
  ugmtOMTFBX->setAxisTitle("BX (Non-zero values prescaled by 107)", 1);

  ugmtOMTFhwPt = ibooker.book1D("ugmtOMTFhwPt", "uGMT OMTF Input p_{T}", 512, -0.5, 511.5);
  ugmtOMTFhwPt->setAxisTitle("Hardware p_{T}", 1);

  ugmtOMTFhwEta = ibooker.book1D("ugmtOMTFhwEta", "uGMT OMTF Input #eta", 231, -115.5, 115.5);
  ugmtOMTFhwEta->setAxisTitle("Hardware #eta", 1);
  
  ugmtOMTFhwPhiPos = ibooker.book1D("ugmtOMTFhwPhiPos", "uGMT OMTF Input #phi, Positive Side", 122, -16.5, 105.5);
  ugmtOMTFhwPhiPos->setAxisTitle("Hardware #phi", 1);

  ugmtOMTFhwPhiNeg = ibooker.book1D("ugmtOMTFhwPhiNeg", "uGMT OMTF Input #phi, Negative Side", 122, -16.5, 105.5);
  ugmtOMTFhwPhiNeg->setAxisTitle("Hardware #phi", 1);

  ugmtOMTFglbPhiPos = ibooker.book1D("ugmtOMTFglbhwPhiPos", "uGMT OMTF Input #phi, Positive Side", 576, -0.5, 575.5);
  ugmtOMTFglbPhiPos->setAxisTitle("Global Hardware #phi", 1);

  ugmtOMTFglbPhiNeg = ibooker.book1D("ugmtOMTFglbhwPhiNeg", "uGMT OMTF Input #phi, Negative Side", 576, -0.5, 575.5);
  ugmtOMTFglbPhiNeg->setAxisTitle("Global Hardware #phi", 1);

  ugmtOMTFhwSign = ibooker.book1D("ugmtOMTFhwSign", "uGMT OMTF Input Sign", 2, -0.5, 1.5);
  ugmtOMTFhwSign->setAxisTitle("Hardware Sign", 1);

  ugmtOMTFhwSignValid = ibooker.book1D("ugmtOMTFhwSignValid", "uGMT OMTF Input SignValid", 2, -0.5, 1.5);
  ugmtOMTFhwSignValid->setAxisTitle("SignValid", 1);

  ugmtOMTFhwQual = ibooker.book1D("ugmtOMTFhwQual", "uGMT OMTF Input Quality", 16, -0.5, 15.5);
  ugmtOMTFhwQual->setAxisTitle("Quality", 1);

  ugmtOMTFlink = ibooker.book1D("ugmtOMTFlink", "uGMT OMTF Input Link", 24, 41.5, 65.5);
  ugmtOMTFlink->setAxisTitle("Link", 1);

  // EMTF Input
  ibooker.setCurrentFolder(monitorDir + "/EMTFInput");

  ugmtEMTFBX = ibooker.book1D("ugmtEMTFBX", "uGMT EMTF Input BX", 5, -2.5, 2.5);
  ugmtEMTFBX->setAxisTitle("BX (Non-zero values prescaled by 107)", 1);

  ugmtEMTFhwPt = ibooker.book1D("ugmtEMTFhwPt", "uGMT EMTF p_{T}", 512, -0.5, 511.5);
  ugmtEMTFhwPt->setAxisTitle("Hardware p_{T}", 1);

  ugmtEMTFhwEta = ibooker.book1D("ugmtEMTFhwEta", "uGMT EMTF #eta", 461, -230.5, 230.5);
  ugmtEMTFhwEta->setAxisTitle("Hardware #eta", 1);
  
  ugmtEMTFhwPhiPos = ibooker.book1D("ugmtEMTFhwPhiPos", "uGMT EMTF #phi, Positive Side", 122, -16.5, 105.5);
  ugmtEMTFhwPhiPos->setAxisTitle("Hardware #phi", 1);

  ugmtEMTFhwPhiNeg = ibooker.book1D("ugmtEMTFhwPhiNeg", "uGMT EMTF #phi, Negative Side", 122, -16.5, 105.5);
  ugmtEMTFhwPhiNeg->setAxisTitle("Hardware #phi", 1);

  ugmtEMTFglbPhiPos = ibooker.book1D("ugmtEMTFglbhwPhiPos", "uGMT EMTF Input Global #phi, Positive Side", 576, -0.5, 575.5);
  ugmtEMTFglbPhiPos->setAxisTitle("Global Hardware #phi", 1);

  ugmtEMTFglbPhiNeg = ibooker.book1D("ugmtEMTFglbhwPhiNeg", "uGMT EMTF Input Global #phi, Negative Side", 576, -0.5, 575.5);
  ugmtEMTFglbPhiNeg->setAxisTitle("Global Hardware #phi", 1);

  ugmtEMTFhwSign = ibooker.book1D("ugmtEMTFhwSign", "uGMT EMTF Sign", 2, -0.5, 1.5);
  ugmtEMTFhwSign->setAxisTitle("Hardware Sign", 1);

  ugmtEMTFhwSignValid = ibooker.book1D("ugmtEMTFhwSignValid", "uGMT EMTF SignValid", 2, -0.5, 1.5);
  ugmtEMTFhwSignValid->setAxisTitle("SignValid", 1);

  ugmtEMTFhwQual = ibooker.book1D("ugmtEMTFhwQual", "uGMT EMTF Quality", 16, -0.5, 15.5);
  ugmtEMTFhwQual->setAxisTitle("Quality", 1);

  ugmtEMTFlink = ibooker.book1D("ugmtEMTFlink", "uGMT EMTF Link", 36, 35.5, 71.5);
  ugmtEMTFlink->setAxisTitle("Link", 1);

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir);

  ugmtBMTFBXvsProcessor = ibooker.book2D("ugmtBMTFBXvsProcessor", "uGMT BMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
  ugmtBMTFBXvsProcessor->setAxisTitle("Wedge", 1);
  for (int bin = 1; bin < 13; ++bin) {
    ugmtBMTFBXvsProcessor->setBinLabel(bin, std::to_string(bin), 1);
  }
  ugmtBMTFBXvsProcessor->setAxisTitle("BX (Non-zero values prescaled by 107)", 2);

  ugmtOMTFBXvsProcessor = ibooker.book2D("ugmtOMTFBXvsProcessor", "uGMT OMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
  ugmtOMTFBXvsProcessor->setAxisTitle("Sector (Detector Side)", 1);
  for (int bin = 1; bin < 7; ++bin) {
    ugmtOMTFBXvsProcessor->setBinLabel(bin, std::to_string(7 - bin) + " (-)", 1);
    ugmtOMTFBXvsProcessor->setBinLabel(bin + 6, std::to_string(bin) + " (+)", 1);
  }
  ugmtOMTFBXvsProcessor->setAxisTitle("BX (Non-zero values prescaled by 107)", 2);

  ugmtEMTFBXvsProcessor = ibooker.book2D("ugmtEMTFBXvsProcessor", "uGMT EMTF Input BX vs Processor", 12, -0.5, 11.5, 5, -2.5, 2.5);
  ugmtEMTFBXvsProcessor->setAxisTitle("Sector (Detector Side)", 1);
  for (int bin = 1; bin < 7; ++bin) {
    ugmtEMTFBXvsProcessor->setBinLabel(bin, std::to_string(7 - bin) + " (-)", 1);
    ugmtEMTFBXvsProcessor->setBinLabel(bin + 6, std::to_string(bin) + " (+)", 1);
  }
  ugmtEMTFBXvsProcessor->setAxisTitle("BX (Non-zero values prescaled by 107)", 2);

  ugmtBXvsLink = ibooker.book2D("ugmtBXvsLink", "uGMT BX vs Input Links", 36, 35.5, 71.5, 5, -2.5, 2.5);
  ugmtBXvsLink->setAxisTitle("Link", 1);
  ugmtBXvsLink->setAxisTitle("BX (Non-zero values prescaled by 107)", 2);
 
  ugmtMuonBX = ibooker.book1D("ugmtMuonBX", "uGMT Muon BX", 5, -2.5, 2.5);
  ugmtMuonBX->setAxisTitle("BX", 1);

  ugmtMuonIndex = ibooker.book1D("ugmtMuonIndex", "uGMT Input Muon Index", 108, -0.5, 107.5);
  ugmtMuonIndex->setAxisTitle("Index", 1);

  ugmtMuonhwPt = ibooker.book1D("ugmtMuonhwPt", "uGMT Muon p_{T}", 512, -0.5, 511.5);
  ugmtMuonhwPt->setAxisTitle("Hardware p_{T}", 1);

  ugmtMuonhwEta = ibooker.book1D("ugmtMuonhwEta", "uGMT Muon #eta", 461, -230.5, 230.5);
  ugmtMuonhwEta->setAxisTitle("Hardware Eta", 1);

  ugmtMuonhwPhi = ibooker.book1D("ugmtMuonhwPhi", "uGMT Muon #phi", 576, -0.5, 575.5);
  ugmtMuonhwPhi->setAxisTitle("Hardware Phi", 1);

  ugmtMuonhwCharge = ibooker.book1D("ugmtMuonhwCharge", "uGMT Muon Charge", 2, -0.5, 1.5);
  ugmtMuonhwCharge->setAxisTitle("Hardware Charge", 1);

  ugmtMuonhwChargeValid = ibooker.book1D("ugmtMuonhwChargeValid", "uGMT Muon ChargeValid", 2, -0.5, 1.5);
  ugmtMuonhwChargeValid->setAxisTitle("ChargeValid", 1);

  ugmtMuonhwQual = ibooker.book1D("ugmtMuonhwQual", "uGMT Muon Quality", 16, -0.5, 15.5);
  ugmtMuonhwQual->setAxisTitle("Quality", 1);

  ugmtMuonhwIso = ibooker.book1D("ugmtMuonhwIso", "uGMT Muon Isolation", 4, -0.5, 3.5);
  ugmtMuonhwIso->setAxisTitle("Isolation", 1);

  ugmtMuonPt = ibooker.book1D("ugmtMuonPt", "uGMT Muon p_{T}", 256, -0.5, 255.5);
  ugmtMuonPt->setAxisTitle("p_{T} [GeV]", 1);

  ugmtMuonEta = ibooker.book1D("ugmtMuonEta", "uGMT Muon #eta", 100, -2.5, 2.5);
  ugmtMuonEta->setAxisTitle("#eta", 1);

  ugmtMuonPhi = ibooker.book1D("ugmtMuonPhi", "uGMT Muon #phi", 126, -3.15, 3.15);
  ugmtMuonPhi->setAxisTitle("#phi", 1);

  ugmtMuonCharge = ibooker.book1D("ugmtMuonCharge", "uGMT Muon Charge", 3, -1.5, 1.5);
  ugmtMuonCharge->setAxisTitle("Charge", 1);

  ugmtMuonPtvsEta = ibooker.book2D("ugmtMuonPtvsEta", "uGMT Muon p_{T} vs #eta", 100, -2.5, 2.5, 256, -0.5, 255.5);
  ugmtMuonPtvsEta->setAxisTitle("#eta", 1);
  ugmtMuonPtvsEta->setAxisTitle("p_{T} [GeV]", 2);

  ugmtMuonPtvsPhi = ibooker.book2D("ugmtMuonPtvsPhi", "uGMT Muon p_{T} vs #phi", 126, -3.15, 3.15, 256, -0.5, 255.5);
  ugmtMuonPtvsPhi->setAxisTitle("#phi", 1);
  ugmtMuonPtvsPhi->setAxisTitle("p_{T} [GeV]", 2);

  ugmtMuonPhivsEta = ibooker.book2D("ugmtMuonPhivsEta", "uGMT Muon #phi vs #eta", 100, -2.5, 2.5, 126, -3.15, 3.15);
  ugmtMuonPhivsEta->setAxisTitle("#eta", 1);
  ugmtMuonPhivsEta->setAxisTitle("#phi", 2);

  ugmtMuonBXvshwPt = ibooker.book2D("ugmtMuonBXvshwPt", "uGMT Muon BX vs p_{T}", 256, -0.5, 511.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwPt->setAxisTitle("Hardware p_{T}", 1);
  ugmtMuonBXvshwPt->setAxisTitle("BX", 2);

  ugmtMuonBXvshwEta = ibooker.book2D("ugmtMuonBXvshwEta", "uGMT Muon BX vs #eta", 93, -232.5, 232.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwEta->setAxisTitle("Hardware #eta", 1);
  ugmtMuonBXvshwEta->setAxisTitle("BX", 2);

  ugmtMuonBXvshwPhi = ibooker.book2D("ugmtMuonBXvshwPhi", "uGMT Muon BX vs #phi", 116, -2.5, 577.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwPhi->setAxisTitle("Hardware #phi", 1);
  ugmtMuonBXvshwPhi->setAxisTitle("BX", 2);

  ugmtMuonBXvshwCharge = ibooker.book2D("ugmtMuonBXvshwCharge", "uGMT Muon BX vs Charge", 2, -0.5, 1.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwCharge->setAxisTitle("Hardware Charge", 1);
  ugmtMuonBXvshwCharge->setAxisTitle("BX", 2);

  ugmtMuonBXvshwChargeValid = ibooker.book2D("ugmtMuonBXvshwChargeValid", "uGMT Muon BX vs ChargeValid", 2, -0.5, 1.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwChargeValid->setAxisTitle("ChargeValid", 1);
  ugmtMuonBXvshwChargeValid->setAxisTitle("BX", 2);

  ugmtMuonBXvshwQual = ibooker.book2D("ugmtMuonBXvshwQual", "uGMT Muon BX vs Quality", 16, -0.5, 15.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwQual->setAxisTitle("Quality", 1);
  ugmtMuonBXvshwQual->setAxisTitle("BX", 2);

  ugmtMuonBXvshwIso = ibooker.book2D("ugmtMuonBXvshwIso", "uGMT Muon BX vs Isolation", 4, -0.5, 3.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwIso->setAxisTitle("Isolation", 1);
  ugmtMuonBXvshwIso->setAxisTitle("BX", 2);
}

void L1TStage2uGMT::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2uGMT") << "L1TStage2uGMT: analyze..." << std::endl;

  edm::Handle<l1t::RegionalMuonCandBxCollection> BMTFBxCollection;
  e.getByToken(ugmtBMTFToken, BMTFBxCollection);

  for (int itBX = BMTFBxCollection->getFirstBX(); itBX <= BMTFBxCollection->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator BMTF = BMTFBxCollection->begin(itBX); BMTF != BMTFBxCollection->end(itBX); ++BMTF) {
      ugmtBMTFBX->Fill(itBX);
      ugmtBMTFhwPt->Fill(BMTF->hwPt());
      ugmtBMTFhwEta->Fill(BMTF->hwEta());
      ugmtBMTFhwPhi->Fill(BMTF->hwPhi());
      ugmtBMTFhwSign->Fill(BMTF->hwSign());
      ugmtBMTFhwSignValid->Fill(BMTF->hwSignValid());
      ugmtBMTFhwQual->Fill(BMTF->hwQual());
      ugmtBMTFlink->Fill(BMTF->link());

      int global_hw_phi = l1t::MicroGMTConfiguration::calcGlobalPhi(BMTF->hwPhi(), BMTF->trackFinderType(), BMTF->processor());
      ugmtBMTFglbPhi->Fill(global_hw_phi);

      ugmtBMTFBXvsProcessor->Fill(BMTF->processor(), itBX);     
      ugmtBXvsLink->Fill(BMTF->link(), itBX);
    }
  }

  edm::Handle<l1t::RegionalMuonCandBxCollection> OMTFBxCollection;
  e.getByToken(ugmtOMTFToken, OMTFBxCollection);

  for (int itBX = OMTFBxCollection->getFirstBX(); itBX <= OMTFBxCollection->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator OMTF = OMTFBxCollection->begin(itBX); OMTF != OMTFBxCollection->end(itBX); ++OMTF) {
      ugmtOMTFBX->Fill(itBX);
      ugmtOMTFhwPt->Fill(OMTF->hwPt());
      ugmtOMTFhwEta->Fill(OMTF->hwEta());
      ugmtOMTFhwSign->Fill(OMTF->hwSign());
      ugmtOMTFhwSignValid->Fill(OMTF->hwSignValid());
      ugmtOMTFhwQual->Fill(OMTF->hwQual());
      ugmtOMTFlink->Fill(OMTF->link());

      int global_hw_phi = l1t::MicroGMTConfiguration::calcGlobalPhi(OMTF->hwPhi(), OMTF->trackFinderType(), OMTF->processor());

      l1t::tftype trackFinderType = OMTF->trackFinderType();

      if (trackFinderType == l1t::omtf_neg) {
        ugmtOMTFBXvsProcessor->Fill(OMTF->processor(), itBX);
        ugmtOMTFhwPhiNeg->Fill(OMTF->hwPhi());
        ugmtOMTFglbPhiNeg->Fill(global_hw_phi);
      } else {
        ugmtOMTFBXvsProcessor->Fill(OMTF->processor() + 6, itBX);
        ugmtOMTFhwPhiPos->Fill(OMTF->hwPhi());
        ugmtOMTFglbPhiPos->Fill(global_hw_phi);
      }

      ugmtBXvsLink->Fill(OMTF->link(), itBX);
    }
  }

  edm::Handle<l1t::RegionalMuonCandBxCollection> EMTFBxCollection;
  e.getByToken(ugmtEMTFToken, EMTFBxCollection);

  for (int itBX = EMTFBxCollection->getFirstBX(); itBX <= EMTFBxCollection->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator EMTF = EMTFBxCollection->begin(itBX); EMTF != EMTFBxCollection->end(itBX); ++EMTF) {
      ugmtEMTFBX->Fill(itBX);
      ugmtEMTFhwPt->Fill(EMTF->hwPt());
      ugmtEMTFhwEta->Fill(EMTF->hwEta());
      ugmtEMTFhwSign->Fill(EMTF->hwSign());
      ugmtEMTFhwSignValid->Fill(EMTF->hwSignValid());
      ugmtEMTFhwQual->Fill(EMTF->hwQual());
      ugmtEMTFlink->Fill(EMTF->link());

      int global_hw_phi = l1t::MicroGMTConfiguration::calcGlobalPhi(EMTF->hwPhi(), EMTF->trackFinderType(), EMTF->processor());

      l1t::tftype trackFinderType = EMTF->trackFinderType();
      
      if (trackFinderType == l1t::emtf_neg) {
        ugmtEMTFBXvsProcessor->Fill(EMTF->processor(), itBX);
        ugmtEMTFhwPhiNeg->Fill(EMTF->hwPhi());
        ugmtEMTFglbPhiNeg->Fill(global_hw_phi);
      } else {
        ugmtEMTFBXvsProcessor->Fill(EMTF->processor() + 6, itBX);
        ugmtEMTFhwPhiPos->Fill(EMTF->hwPhi());
        ugmtEMTFglbPhiPos->Fill(global_hw_phi);
      }

      ugmtBXvsLink->Fill(EMTF->link(), itBX);
    }
  }

  edm::Handle<l1t::MuonBxCollection> MuonBxCollection;
  e.getByToken(ugmtMuonToken, MuonBxCollection);

  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::MuonBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {
      ugmtMuonBX->Fill(itBX);
      ugmtMuonIndex->Fill(Muon->tfMuonIndex());
      ugmtMuonhwPt->Fill(Muon->hwPt());
      ugmtMuonhwEta->Fill(Muon->hwEta());
      ugmtMuonhwPhi->Fill(Muon->hwPhi());
      ugmtMuonhwCharge->Fill(Muon->hwCharge());
      ugmtMuonhwChargeValid->Fill(Muon->hwChargeValid());
      ugmtMuonhwQual->Fill(Muon->hwQual());
      ugmtMuonhwIso->Fill(Muon->hwIso());

      ugmtMuonPt->Fill(Muon->pt());
      ugmtMuonEta->Fill(Muon->eta());
      ugmtMuonPhi->Fill(Muon->phi());
      ugmtMuonCharge->Fill(Muon->charge());

      ugmtMuonPtvsEta->Fill(Muon->eta(), Muon->pt());
      ugmtMuonPtvsPhi->Fill(Muon->phi(), Muon->pt());
      ugmtMuonPhivsEta->Fill(Muon->eta(), Muon->phi());

      ugmtMuonBXvshwPt->Fill(Muon->hwPt(), itBX);
      ugmtMuonBXvshwEta->Fill(Muon->hwEta(), itBX);
      ugmtMuonBXvshwPhi->Fill(Muon->hwPhi(), itBX);
      ugmtMuonBXvshwCharge->Fill(Muon->hwCharge(), itBX);
      ugmtMuonBXvshwChargeValid->Fill(Muon->hwChargeValid(), itBX);
      ugmtMuonBXvshwQual->Fill(Muon->hwQual(), itBX);
      ugmtMuonBXvshwIso->Fill(Muon->hwIso(), itBX);
    }
  }
}

