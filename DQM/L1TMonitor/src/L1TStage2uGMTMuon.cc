#include "DQM/L1TMonitor/interface/L1TStage2uGMTMuon.h"


L1TStage2uGMTMuon::L1TStage2uGMTMuon(const edm::ParameterSet& ps)
    : ugmtMuonToken(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonProducer"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir")),
      titlePrefix(ps.getUntrackedParameter<std::string>("titlePrefix")),
      verbose(ps.getUntrackedParameter<bool>("verbose")),
      makeMuonAtVtxPlots(ps.getUntrackedParameter<bool>("makeMuonAtVtxPlots"))
{
}

L1TStage2uGMTMuon::~L1TStage2uGMTMuon() {}

void L1TStage2uGMTMuon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonProducer");
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<std::string>("titlePrefix", "")->setComment("Prefix text for the histogram titles.");
  desc.addUntracked<bool>("verbose", false);
  desc.addUntracked<bool>("makeMuonAtVtxPlots", false);
  descriptions.add("l1tStage2uGMTMuon", desc);
}

void L1TStage2uGMTMuon::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c, ugmtmuondqm::Histograms& histograms) const {}

void L1TStage2uGMTMuon::bookHistograms(DQMStore::ConcurrentBooker& booker, const edm::Run&, const edm::EventSetup&, ugmtmuondqm::Histograms& histograms) const {

  // Subsystem Monitoring and Muon Output
  booker.setCurrentFolder(monitorDir);

  histograms.ugmtMuonBX = booker.book1D("ugmtMuonBX", titlePrefix+"BX", 7, -3.5, 3.5);
  histograms.ugmtMuonBX.setAxisTitle("BX", 1);

  histograms.ugmtnMuons = booker.book1D("ugmtnMuons", titlePrefix+"Multiplicity", 9, -0.5, 8.5);
  histograms.ugmtnMuons.setAxisTitle("Muon Multiplicity (BX == 0)", 1);

  histograms.ugmtMuonhwPt = booker.book1D("ugmtMuonhwPt", titlePrefix+"p_{T}", 512, -0.5, 511.5);
  histograms.ugmtMuonhwPt.setAxisTitle("Hardware p_{T}", 1);

  histograms.ugmtMuonhwEta = booker.book1D("ugmtMuonhwEta", titlePrefix+"#eta", 461, -230.5, 230.5);
  histograms.ugmtMuonhwEta.setAxisTitle("Hardware Eta", 1);

  histograms.ugmtMuonhwPhi = booker.book1D("ugmtMuonhwPhi", titlePrefix+"#phi", 576, -0.5, 575.5);
  histograms.ugmtMuonhwPhi.setAxisTitle("Hardware Phi", 1);

  histograms.ugmtMuonhwCharge = booker.book1D("ugmtMuonhwCharge", titlePrefix+"Charge", 2, -0.5, 1.5);
  histograms.ugmtMuonhwCharge.setAxisTitle("Hardware Charge", 1);

  histograms.ugmtMuonhwChargeValid = booker.book1D("ugmtMuonhwChargeValid", titlePrefix+"ChargeValid", 2, -0.5, 1.5);
  histograms.ugmtMuonhwChargeValid.setAxisTitle("ChargeValid", 1);

  histograms.ugmtMuonhwQual = booker.book1D("ugmtMuonhwQual", titlePrefix+"Quality", 16, -0.5, 15.5);
  histograms.ugmtMuonhwQual.setAxisTitle("Quality", 1);

  histograms.ugmtMuonPt = booker.book1D("ugmtMuonPt", titlePrefix+"p_{T}", 256, -0.5, 255.5);
  histograms.ugmtMuonPt.setAxisTitle("p_{T} [GeV]", 1);

  histograms.ugmtMuonEta = booker.book1D("ugmtMuonEta", titlePrefix+"#eta", 100, -2.5, 2.5);
  histograms.ugmtMuonEta.setAxisTitle("#eta", 1);

  histograms.ugmtMuonPhi = booker.book1D("ugmtMuonPhi", titlePrefix+"#phi", 126, -3.15, 3.15);
  histograms.ugmtMuonPhi.setAxisTitle("#phi", 1);

  histograms.ugmtMuonCharge = booker.book1D("ugmtMuonCharge", titlePrefix+"Charge", 3, -1.5, 1.5);
  histograms.ugmtMuonCharge.setAxisTitle("Charge", 1);

  histograms.ugmtMuonPtvsEta = booker.book2D("ugmtMuonPtvsEta", titlePrefix+"p_{T} vs #eta", 100, -2.5, 2.5, 256, -0.5, 255.5);
  histograms.ugmtMuonPtvsEta.setAxisTitle("#eta", 1);
  histograms.ugmtMuonPtvsEta.setAxisTitle("p_{T} [GeV]", 2);

  histograms.ugmtMuonPtvsPhi = booker.book2D("ugmtMuonPtvsPhi", titlePrefix+"p_{T} vs #phi", 64, -3.2, 3.2, 256, -0.5, 255.5);
  histograms.ugmtMuonPtvsPhi.setAxisTitle("#phi", 1);
  histograms.ugmtMuonPtvsPhi.setAxisTitle("p_{T} [GeV]", 2);

  histograms.ugmtMuonPhivsEta = booker.book2D("ugmtMuonPhivsEta", titlePrefix+"#phi vs #eta", 100, -2.5, 2.5, 64, -3.2, 3.2);
  histograms.ugmtMuonPhivsEta.setAxisTitle("#eta", 1);
  histograms.ugmtMuonPhivsEta.setAxisTitle("#phi", 2);

  histograms.ugmtMuonBXvshwPt = booker.book2D("ugmtMuonBXvshwPt", titlePrefix+"BX vs p_{T}", 256, -0.5, 511.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwPt.setAxisTitle("Hardware p_{T}", 1);
  histograms.ugmtMuonBXvshwPt.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwEta = booker.book2D("ugmtMuonBXvshwEta", titlePrefix+"BX vs #eta", 93, -232.5, 232.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwEta.setAxisTitle("Hardware #eta", 1);
  histograms.ugmtMuonBXvshwEta.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwPhi = booker.book2D("ugmtMuonBXvshwPhi", titlePrefix+"BX vs #phi", 116, -2.5, 577.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwPhi.setAxisTitle("Hardware #phi", 1);
  histograms.ugmtMuonBXvshwPhi.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwCharge = booker.book2D("ugmtMuonBXvshwCharge", titlePrefix+"BX vs Charge", 2, -0.5, 1.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwCharge.setAxisTitle("Hardware Charge", 1);
  histograms.ugmtMuonBXvshwCharge.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwChargeValid = booker.book2D("ugmtMuonBXvshwChargeValid", titlePrefix+"BX vs ChargeValid", 2, -0.5, 1.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwChargeValid.setAxisTitle("ChargeValid", 1);
  histograms.ugmtMuonBXvshwChargeValid.setAxisTitle("BX", 2);

  histograms.ugmtMuonBXvshwQual = booker.book2D("ugmtMuonBXvshwQual", titlePrefix+"BX vs Quality", 16, -0.5, 15.5, 5, -2.5, 2.5);
  histograms.ugmtMuonBXvshwQual.setAxisTitle("Quality", 1);
  histograms.ugmtMuonBXvshwQual.setAxisTitle("BX", 2);

  if (makeMuonAtVtxPlots) {
    histograms.ugmtMuonhwEtaAtVtx = booker.book1D("ugmtMuonhwEtaAtVtx", titlePrefix+"#eta at vertex", 461, -230.5, 230.5);
    histograms.ugmtMuonhwEtaAtVtx.setAxisTitle("Hardware Eta at Vertex", 1);

    histograms.ugmtMuonhwPhiAtVtx = booker.book1D("ugmtMuonhwPhiAtVtx", titlePrefix+"#phi at vertex", 576, -0.5, 575.5);
    histograms.ugmtMuonhwPhiAtVtx.setAxisTitle("Hardware Phi at Vertex", 1);

    histograms.ugmtMuonEtaAtVtx = booker.book1D("ugmtMuonEtaAtVtx", titlePrefix+"#eta at vertex", 100, -2.5, 2.5);
    histograms.ugmtMuonEtaAtVtx.setAxisTitle("#eta at Vertex", 1);

    histograms.ugmtMuonPhiAtVtx = booker.book1D("ugmtMuonPhiAtVtx", titlePrefix+"#phi at vertex", 126, -3.15, 3.15);
    histograms.ugmtMuonPhiAtVtx.setAxisTitle("#phi at Vertex", 1);

    histograms.ugmtMuonPtvsEtaAtVtx = booker.book2D("ugmtMuonPtvsEtaAtVtx", titlePrefix+"p_{T} vs #eta at vertex", 100, -2.5, 2.5, 256, -0.5, 255.5);
    histograms.ugmtMuonPtvsEtaAtVtx.setAxisTitle("#eta at Vertex", 1);
    histograms.ugmtMuonPtvsEtaAtVtx.setAxisTitle("p_{T} [GeV]", 2);

    histograms.ugmtMuonPhiAtVtxvsEtaAtVtx = booker.book2D("ugmtMuonPhiAtVtxvsEtaAtVtx", titlePrefix+"#phi_{vtx} vs #eta_{vtx}", 100, -2.5, 2.5, 64, -3.2, 3.2);
    histograms.ugmtMuonPhiAtVtxvsEtaAtVtx.setAxisTitle("#eta at Vertex", 1);
    histograms.ugmtMuonPhiAtVtxvsEtaAtVtx.setAxisTitle("#phi at Vertex", 2);

    histograms.ugmtMuonPtvsPhiAtVtx = booker.book2D("ugmtMuonPtvsPhiAtVtx", titlePrefix+"p_{T} vs #phi at vertex", 64, -3.2, 3.2, 256, -0.5, 255.5);
    histograms.ugmtMuonPtvsPhiAtVtx.setAxisTitle("#phi at Vertex", 1);
    histograms.ugmtMuonPtvsPhiAtVtx.setAxisTitle("p_{T} [GeV]", 2);

    histograms.ugmtMuonBXvshwEtaAtVtx = booker.book2D("ugmtMuonBXvshwEtaAtVtx", titlePrefix+"BX vs #eta at vertex", 93, -232.5, 232.5, 5, -2.5, 2.5);
    histograms.ugmtMuonBXvshwEtaAtVtx.setAxisTitle("Hardware #eta at Vertex", 1);
    histograms.ugmtMuonBXvshwEtaAtVtx.setAxisTitle("BX", 2);

    histograms.ugmtMuonBXvshwPhiAtVtx = booker.book2D("ugmtMuonBXvshwPhiAtVtx", titlePrefix+"BX vs #phi at vertex", 116, -2.5, 577.5, 5, -2.5, 2.5);
    histograms.ugmtMuonBXvshwPhiAtVtx.setAxisTitle("Hardware #phi at Vertex", 1);
    histograms.ugmtMuonBXvshwPhiAtVtx.setAxisTitle("BX", 2);
  }
}

void L1TStage2uGMTMuon::dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, ugmtmuondqm::Histograms const& histograms) const {

  if (verbose) edm::LogInfo("L1TStage2uGMTMuon") << "L1TStage2uGMTMuon: analyze..." << std::endl;

  edm::Handle<l1t::MuonBxCollection> MuonBxCollection;
  e.getByToken(ugmtMuonToken, MuonBxCollection);

  histograms.ugmtnMuons.fill(MuonBxCollection->size(0));

  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::MuonBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {

      histograms.ugmtMuonBX.fill(itBX);
      histograms.ugmtMuonhwPt.fill(Muon->hwPt());
      histograms.ugmtMuonhwEta.fill(Muon->hwEta());
      histograms.ugmtMuonhwPhi.fill(Muon->hwPhi());
      histograms.ugmtMuonhwCharge.fill(Muon->hwCharge());
      histograms.ugmtMuonhwChargeValid.fill(Muon->hwChargeValid());
      histograms.ugmtMuonhwQual.fill(Muon->hwQual());

      histograms.ugmtMuonPt.fill(Muon->pt());
      histograms.ugmtMuonEta.fill(Muon->eta());
      histograms.ugmtMuonPhi.fill(Muon->phi());
      histograms.ugmtMuonCharge.fill(Muon->charge());

      histograms.ugmtMuonPtvsEta.fill(Muon->eta(), Muon->pt());
      histograms.ugmtMuonPtvsPhi.fill(Muon->phi(), Muon->pt());
      histograms.ugmtMuonPhivsEta.fill(Muon->eta(), Muon->phi());

      histograms.ugmtMuonBXvshwPt.fill(Muon->hwPt(), itBX);
      histograms.ugmtMuonBXvshwEta.fill(Muon->hwEta(), itBX);
      histograms.ugmtMuonBXvshwPhi.fill(Muon->hwPhi(), itBX);
      histograms.ugmtMuonBXvshwCharge.fill(Muon->hwCharge(), itBX);
      histograms.ugmtMuonBXvshwChargeValid.fill(Muon->hwChargeValid(), itBX);
      histograms.ugmtMuonBXvshwQual.fill(Muon->hwQual(), itBX);

      if (makeMuonAtVtxPlots) {
        histograms.ugmtMuonhwEtaAtVtx.fill(Muon->hwEtaAtVtx());
        histograms.ugmtMuonhwPhiAtVtx.fill(Muon->hwPhiAtVtx());
        histograms.ugmtMuonEtaAtVtx.fill(Muon->etaAtVtx());
        histograms.ugmtMuonPhiAtVtx.fill(Muon->phiAtVtx());
        histograms.ugmtMuonPtvsEtaAtVtx.fill(Muon->etaAtVtx(), Muon->pt());
        histograms.ugmtMuonPtvsPhiAtVtx.fill(Muon->phiAtVtx(), Muon->pt());
        histograms.ugmtMuonPhiAtVtxvsEtaAtVtx.fill(Muon->etaAtVtx(), Muon->phiAtVtx());
        histograms.ugmtMuonBXvshwEtaAtVtx.fill(Muon->hwEtaAtVtx(), itBX);
        histograms.ugmtMuonBXvshwPhiAtVtx.fill(Muon->hwPhiAtVtx(), itBX);
      }
    }
  }
}

