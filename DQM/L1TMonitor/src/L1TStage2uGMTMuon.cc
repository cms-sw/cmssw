#include "DQM/L1TMonitor/interface/L1TStage2uGMTMuon.h"


L1TStage2uGMTMuon::L1TStage2uGMTMuon(const edm::ParameterSet& ps)
    : ugmtMuonToken(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonProducer"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      titlePrefix(ps.getUntrackedParameter<std::string>("titlePrefix", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false))
{
}

L1TStage2uGMTMuon::~L1TStage2uGMTMuon() {}

void L1TStage2uGMTMuon::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2uGMTMuon::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2uGMTMuon::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir);

  ugmtMuonBX = ibooker.book1D("ugmtMuonBX", (titlePrefix+"BX").c_str(), 5, -2.5, 2.5);
  ugmtMuonBX->setAxisTitle("BX", 1);

  ugmtnMuons = ibooker.book1D("ugmtnMuons", (titlePrefix+"Multiplicity").c_str(), 9, -0.5, 8.5);
  ugmtnMuons->setAxisTitle("Muon Multiplicity (BX == 0)", 1);

  ugmtMuonhwPt = ibooker.book1D("ugmtMuonhwPt", (titlePrefix+"p_{T}").c_str(), 512, -0.5, 511.5);
  ugmtMuonhwPt->setAxisTitle("Hardware p_{T}", 1);

  ugmtMuonhwEta = ibooker.book1D("ugmtMuonhwEta", (titlePrefix+"#eta").c_str(), 461, -230.5, 230.5);
  ugmtMuonhwEta->setAxisTitle("Hardware Eta", 1);

  ugmtMuonhwPhi = ibooker.book1D("ugmtMuonhwPhi", (titlePrefix+"#phi").c_str(), 576, -0.5, 575.5);
  ugmtMuonhwPhi->setAxisTitle("Hardware Phi", 1);

  ugmtMuonhwCharge = ibooker.book1D("ugmtMuonhwCharge", (titlePrefix+"Charge").c_str(), 2, -0.5, 1.5);
  ugmtMuonhwCharge->setAxisTitle("Hardware Charge", 1);

  ugmtMuonhwChargeValid = ibooker.book1D("ugmtMuonhwChargeValid", (titlePrefix+"ChargeValid").c_str(), 2, -0.5, 1.5);
  ugmtMuonhwChargeValid->setAxisTitle("ChargeValid", 1);

  ugmtMuonhwQual = ibooker.book1D("ugmtMuonhwQual", (titlePrefix+"Quality").c_str(), 16, -0.5, 15.5);
  ugmtMuonhwQual->setAxisTitle("Quality", 1);

  ugmtMuonPt = ibooker.book1D("ugmtMuonPt", (titlePrefix+"p_{T}").c_str(), 256, -0.5, 255.5);
  ugmtMuonPt->setAxisTitle("p_{T} [GeV]", 1);

  ugmtMuonEta = ibooker.book1D("ugmtMuonEta", (titlePrefix+"#eta").c_str(), 100, -2.5, 2.5);
  ugmtMuonEta->setAxisTitle("#eta", 1);

  ugmtMuonPhi = ibooker.book1D("ugmtMuonPhi", (titlePrefix+"#phi").c_str(), 126, -3.15, 3.15);
  ugmtMuonPhi->setAxisTitle("#phi", 1);

  ugmtMuonCharge = ibooker.book1D("ugmtMuonCharge", (titlePrefix+"Charge").c_str(), 3, -1.5, 1.5);
  ugmtMuonCharge->setAxisTitle("Charge", 1);

  ugmtMuonPtvsEta = ibooker.book2D("ugmtMuonPtvsEta", (titlePrefix+"p_{T} vs #eta").c_str(), 100, -2.5, 2.5, 256, -0.5, 255.5);
  ugmtMuonPtvsEta->setAxisTitle("#eta", 1);
  ugmtMuonPtvsEta->setAxisTitle("p_{T} [GeV]", 2);

  ugmtMuonPtvsPhi = ibooker.book2D("ugmtMuonPtvsPhi", (titlePrefix+"p_{T} vs #phi").c_str(), 64, -3.2, 3.2, 256, -0.5, 255.5);
  ugmtMuonPtvsPhi->setAxisTitle("#phi", 1);
  ugmtMuonPtvsPhi->setAxisTitle("p_{T} [GeV]", 2);

  ugmtMuonPhivsEta = ibooker.book2D("ugmtMuonPhivsEta", (titlePrefix+"#phi vs #eta").c_str(), 100, -2.5, 2.5, 64, -3.2, 3.2);
  ugmtMuonPhivsEta->setAxisTitle("#eta", 1);
  ugmtMuonPhivsEta->setAxisTitle("#phi", 2);

  ugmtMuonBXvshwPt = ibooker.book2D("ugmtMuonBXvshwPt", (titlePrefix+"BX vs p_{T}").c_str(), 256, -0.5, 511.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwPt->setAxisTitle("Hardware p_{T}", 1);
  ugmtMuonBXvshwPt->setAxisTitle("BX", 2);

  ugmtMuonBXvshwEta = ibooker.book2D("ugmtMuonBXvshwEta", (titlePrefix+"BX vs #eta").c_str(), 93, -232.5, 232.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwEta->setAxisTitle("Hardware #eta", 1);
  ugmtMuonBXvshwEta->setAxisTitle("BX", 2);

  ugmtMuonBXvshwPhi = ibooker.book2D("ugmtMuonBXvshwPhi", (titlePrefix+"BX vs #phi").c_str(), 116, -2.5, 577.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwPhi->setAxisTitle("Hardware #phi", 1);
  ugmtMuonBXvshwPhi->setAxisTitle("BX", 2);

  ugmtMuonBXvshwCharge = ibooker.book2D("ugmtMuonBXvshwCharge", (titlePrefix+"BX vs Charge").c_str(), 2, -0.5, 1.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwCharge->setAxisTitle("Hardware Charge", 1);
  ugmtMuonBXvshwCharge->setAxisTitle("BX", 2);

  ugmtMuonBXvshwChargeValid = ibooker.book2D("ugmtMuonBXvshwChargeValid", (titlePrefix+"BX vs ChargeValid").c_str(), 2, -0.5, 1.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwChargeValid->setAxisTitle("ChargeValid", 1);
  ugmtMuonBXvshwChargeValid->setAxisTitle("BX", 2);

  ugmtMuonBXvshwQual = ibooker.book2D("ugmtMuonBXvshwQual", (titlePrefix+"BX vs Quality").c_str(), 16, -0.5, 15.5, 5, -2.5, 2.5);
  ugmtMuonBXvshwQual->setAxisTitle("Quality", 1);
  ugmtMuonBXvshwQual->setAxisTitle("BX", 2);
}

void L1TStage2uGMTMuon::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2uGMTMuon") << "L1TStage2uGMTMuon: analyze..." << std::endl;

  edm::Handle<l1t::MuonBxCollection> MuonBxCollection;
  e.getByToken(ugmtMuonToken, MuonBxCollection);

  ugmtnMuons->Fill(MuonBxCollection->size(0));

  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::MuonBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {

      ugmtMuonBX->Fill(itBX);
      ugmtMuonhwPt->Fill(Muon->hwPt());
      ugmtMuonhwEta->Fill(Muon->hwEta());
      ugmtMuonhwPhi->Fill(Muon->hwPhi());
      ugmtMuonhwCharge->Fill(Muon->hwCharge());
      ugmtMuonhwChargeValid->Fill(Muon->hwChargeValid());
      ugmtMuonhwQual->Fill(Muon->hwQual());

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
    }
  }
}

