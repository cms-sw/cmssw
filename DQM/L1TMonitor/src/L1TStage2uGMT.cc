#include "DQM/L1TMonitor/interface/L1TStage2uGMT.h"


L1TStage2uGMT::L1TStage2uGMT(const edm::ParameterSet& ps)
    : ugmtToken(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("ugmtSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2uGMT::~L1TStage2uGMT() {}

void L1TStage2uGMT::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2uGMT::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2uGMT::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  ibooker.setCurrentFolder(monitorDir);

  ugmtBX = ibooker.book1D("ugmtBX", "uGMT BX", 5, -2.5, 2.5);
  ugmtPt = ibooker.book1D("ugmtPt", "uGMT p_{T}", 511, -0.5, 510.5);
  ugmtEta = ibooker.book1D("ugmtEta", "uGMT #eta", 447, -223.5, 223.5);
  ugmtPhi = ibooker.book1D("ugmtPhi", "uGMT #phi", 576, -0.5, 575.5);
  ugmtCharge = ibooker.book1D("ugmtCharge", "uGMT Charge", 2, -0.5, 1.5);
  ugmtChargeValid = ibooker.book1D("ugmtChargeValid", "uGMT ChargeValid", 2, -0.5, 1.5);
  ugmtQual = ibooker.book1D("ugmtQual", "uGMT Quality", 21, -0.5, 20.5);
  ugmtIso = ibooker.book1D("ugmtIso", "uGMT Isolation", 5, -0.5, 4.5);

  ugmtBXvsPt = ibooker.book2D("ugmtBXvsPt", "uGMT BX vs p_{T}", 511, -0.5, 510.5, 5, -2.5, 2.5);
  ugmtBXvsEta = ibooker.book2D("ugmtBXvsEta", "uGMT BX vs #eta", 447, -223.5, 223.5, 5, -2.5, 2.5);
  ugmtBXvsPhi = ibooker.book2D("ugmtBXvsPhi", "uGMT BX vs #phi", 576, -0.5, 575.5, 5, -2.5, 2.5);
  ugmtBXvsCharge = ibooker.book2D("ugmtBXvsCharge", "uGMT BX vs Charge", 2, -0.5, 1.5, 5, -2.5, 2.5);
  ugmtBXvsChargeValid = ibooker.book2D("ugmtBXvsChargeValid", "uGMT BX vs ChargeValid", 2, -0.5, 1.5, 5, -2.5, 2.5);
  ugmtBXvsQual = ibooker.book2D("ugmtBXvsQual", "uGMT BX vs Quality", 21, -0.5, 20.5, 5, -2.5, 2.5);
  ugmtBXvsIso = ibooker.book2D("ugmtBXvsIso", "uGMT BX vs Isolation", 5, -0.5, 4.5, 5, -2.5, 2.5);

  ugmtPtvsEta = ibooker.book2D("ugmtPtvsEta", "uGMT p_{T} vs #eta", 447, -223.5, 223.5, 511, -0.5, 510.5);
  ugmtPtvsPhi = ibooker.book2D("ugmtPtvsPhi", "uGMT p_{T} vs #phi", 576, -0.5, 575.5, 511, -0.5, 510.5);
  ugmtPhivsEta = ibooker.book2D("ugmtPhivsEta", "uGMT #phi vs #eta", 447, -223.5, 223.5, 576, -0.5, 575.5);
}

void L1TStage2uGMT::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2uGMT") << "L1TStage2uGMT: analyze..." << std::endl;

  edm::Handle<l1t::MuonBxCollection> MuonBxCollection;
  e.getByToken(ugmtToken, MuonBxCollection);

  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::MuonBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {
      int hwPt = Muon->hwPt();
      int hwEta = Muon->hwEta();
      int hwPhi = Muon->hwPhi();
      int hwCharge = Muon->hwCharge();
      int hwChargeValid = Muon->hwChargeValid();
      int hwQual = Muon->hwQual();
      int hwIso = Muon->hwIso();

      ugmtBX->Fill(itBX);
      ugmtPt->Fill(hwPt);
      ugmtEta->Fill(hwEta);
      ugmtPhi->Fill(hwPhi);
      ugmtCharge->Fill(hwCharge);
      ugmtChargeValid->Fill(hwChargeValid);
      ugmtQual->Fill(hwQual);
      ugmtIso->Fill(hwIso);

      ugmtBXvsPt->Fill(hwPt, itBX);
      ugmtBXvsEta->Fill(hwEta, itBX);
      ugmtBXvsPhi->Fill(hwPhi, itBX);
      ugmtBXvsCharge->Fill(hwCharge, itBX);
      ugmtBXvsChargeValid->Fill(hwChargeValid, itBX);
      ugmtBXvsQual->Fill(hwQual, itBX);
      ugmtBXvsIso->Fill(hwIso, itBX);
      
      ugmtPtvsEta->Fill(hwEta, hwPt); 
      ugmtPtvsPhi->Fill(hwPhi, hwPt); 
      ugmtPhivsEta->Fill(hwEta, hwPhi);
    }
  }
}

