#include "DQM/L1TMonitor/interface/L1TStage2MuonComp.h"


L1TStage2MuonComp::L1TStage2MuonComp(const edm::ParameterSet& ps)
    : muonToken1(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection1"))),
      muonToken2(consumes<l1t::MuonBxCollection>(ps.getParameter<edm::InputTag>("muonCollection2"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir")),
      muonColl1Title(ps.getUntrackedParameter<std::string>("muonCollection1Title")),
      muonColl2Title(ps.getUntrackedParameter<std::string>("muonCollection2Title")),
      summaryTitle(ps.getUntrackedParameter<std::string>("summaryTitle")),
      ignoreBin(ps.getUntrackedParameter<std::vector<int>>("ignoreBin")),
      verbose(ps.getUntrackedParameter<bool>("verbose"))
{
  // First include all bins
  for (unsigned int i = 1; i <= RIDX; i++) {
    incBin[i] = true;
  }
  // Then check the list of bins to ignore
  for (const auto& i : ignoreBin) {
    if (i > 0 && i <= RIDX) {
      incBin[i] = false;
    }
  }
}

L1TStage2MuonComp::~L1TStage2MuonComp() {}

void L1TStage2MuonComp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonCollection1")->setComment("L1T Muon collection 1");
  desc.add<edm::InputTag>("muonCollection2")->setComment("L1T Muon collection 2");
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<std::string>("muonCollection1Title", "Muon collection 1")->setComment("Histogram title for first collection.");
  desc.addUntracked<std::string>("muonCollection2Title", "Muon collection 2")->setComment("Histogram title for second collection.");
  desc.addUntracked<std::string>("summaryTitle", "Summary")->setComment("Title of summary histogram.");
  desc.addUntracked<std::vector<int>>("ignoreBin", std::vector<int>())->setComment("List of bins to ignore");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("l1tStage2MuonComp", desc);
}

void L1TStage2MuonComp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c, muoncompdqm::Histograms& histograms) const
{}

void L1TStage2MuonComp::bookHistograms(DQMStore::ConcurrentBooker& booker, const edm::Run&, const edm::EventSetup&, muoncompdqm::Histograms& histograms) const
{

  // Subsystem Monitoring and Muon Output
  booker.setCurrentFolder(monitorDir);

  histograms.summary = booker.book1D("summary", summaryTitle.c_str(), 16, 1, 17); // range to match bin numbering
  histograms.summary.setBinLabel(BXRANGEGOOD, "BX range match", 1);
  histograms.summary.setBinLabel(BXRANGEBAD, "BX range mismatch", 1);
  histograms.summary.setBinLabel(NMUONGOOD, "muon collection size match", 1);
  histograms.summary.setBinLabel(NMUONBAD, "muon collection size mismatch", 1);
  histograms.summary.setBinLabel(MUONALL, "# muons", 1);
  histograms.summary.setBinLabel(MUONGOOD, "# matching muons", 1);
  histograms.summary.setBinLabel(PTBAD, "p_{T} mismatch", 1);
  histograms.summary.setBinLabel(ETABAD, "#eta mismatch", 1);
  histograms.summary.setBinLabel(PHIBAD, "#phi mismatch", 1);
  histograms.summary.setBinLabel(ETAATVTXBAD, "#eta at vertex mismatch", 1);
  histograms.summary.setBinLabel(PHIATVTXBAD, "#phi at vertex mismatch", 1);
  histograms.summary.setBinLabel(CHARGEBAD, "charge mismatch", 1);
  histograms.summary.setBinLabel(CHARGEVALBAD, "charge valid mismatch", 1);
  histograms.summary.setBinLabel(QUALBAD, "quality mismatch", 1);
  histograms.summary.setBinLabel(ISOBAD, "iso mismatch", 1);
  histograms.summary.setBinLabel(IDXBAD, "index mismatch", 1);

  histograms.errorSummaryNum = booker.book1D("errorSummaryNum", summaryTitle.c_str(), 13, 1, 14); // range to match bin numbering
  histograms.errorSummaryNum.setBinLabel(RBXRANGE, "BX range mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RNMUON, "muon collection size mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RMUON, "mismatching muons", 1);
  histograms.errorSummaryNum.setBinLabel(RPT, "p_{T} mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RETA, "#eta mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RPHI, "#phi mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RETAATVTX, "#eta at vertex mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RPHIATVTX, "#phi at vertex mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RCHARGE, "charge mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RCHARGEVAL, "charge valid mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RQUAL, "quality mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RISO, "iso mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RIDX, "index mismatch", 1);

  // Change the label for those bins that will be ignored
  for (unsigned int i = 1; i <= RIDX; i++) {
    if (incBin[i]==false) {
      histograms.errorSummaryNum.setBinLabel(i, "Ignored", 1);
    }
  }

  histograms.errorSummaryDen = booker.book1D("errorSummaryDen", "denominators", 13, 1, 14); // range to match bin numbering
  histograms.errorSummaryDen.setBinLabel(RBXRANGE, "# events", 1);
  histograms.errorSummaryDen.setBinLabel(RNMUON, "# muon collections", 1);
  for (int i = RMUON; i <= RIDX; ++i) {
    histograms.errorSummaryDen.setBinLabel(i, "# muons", 1);
  }

  histograms.muColl1BxRange = booker.book1D("muBxRangeColl1", (muonColl1Title+" mismatching BX range").c_str(), 5, -2.5, 2.5);
  histograms.muColl1BxRange.setAxisTitle("BX range", 1);
  histograms.muColl1nMu = booker.book1D("nMuColl1", (muonColl1Title+" mismatching muon multiplicity").c_str(), 9, -0.5, 8.5);
  histograms.muColl1nMu.setAxisTitle("Muon multiplicity", 1);
  histograms.muColl1hwPt = booker.book1D("muHwPtColl1", (muonColl1Title+" mismatching muon p_{T}").c_str(), 512, -0.5, 511.5);
  histograms.muColl1hwPt.setAxisTitle("Hardware p_{T}", 1);
  histograms.muColl1hwEta = booker.book1D("muHwEtaColl1", (muonColl1Title+" mismatching muon #eta").c_str(), 461, -230.5, 230.5);
  histograms.muColl1hwEta.setAxisTitle("Hardware #eta", 1);
  histograms.muColl1hwPhi = booker.book1D("muHwPhiColl1", (muonColl1Title+" mismatching muon #phi").c_str(), 576, -0.5, 575.5);
  histograms.muColl1hwPhi.setAxisTitle("Hardware #phi", 1);
  histograms.muColl1hwEtaAtVtx = booker.book1D("muHwEtaAtVtxColl1", (muonColl1Title+" mismatching muon #eta at vertex").c_str(), 461, -230.5, 230.5);
  histograms.muColl1hwEtaAtVtx.setAxisTitle("Hardware #eta at vertex", 1);
  histograms.muColl1hwPhiAtVtx = booker.book1D("muHwPhiAtVtxColl1", (muonColl1Title+" mismatching muon #phi at vertex").c_str(), 576, -0.5, 575.5);
  histograms.muColl1hwPhiAtVtx.setAxisTitle("Hardware #phi at vertex", 1);
  histograms.muColl1hwCharge = booker.book1D("muHwChargeColl1", (muonColl1Title+" mismatching muon charge").c_str(), 2, -0.5, 1.5);
  histograms.muColl1hwCharge.setAxisTitle("Hardware charge", 1);
  histograms.muColl1hwChargeValid = booker.book1D("muHwChargeValidColl1", (muonColl1Title+" mismatching muon charge valid").c_str(), 2, -0.5, 1.5);
  histograms.muColl1hwChargeValid.setAxisTitle("Hardware charge valid", 1);
  histograms.muColl1hwQual = booker.book1D("muHwQualColl1", (muonColl1Title+" mismatching muon quality").c_str(), 16, -0.5, 15.5);
  histograms.muColl1hwQual.setAxisTitle("Hardware quality", 1);
  histograms.muColl1hwIso = booker.book1D("muHwIsoColl1", (muonColl1Title+" mismatching muon isolation").c_str(), 4, -0.5, 3.5);
  histograms.muColl1hwIso.setAxisTitle("Hardware isolation", 1);
  histograms.muColl1Index = booker.book1D("muIndexColl1", (muonColl1Title+" mismatching Input muon index").c_str(), 108, -0.5, 107.5);
  histograms.muColl1Index.setAxisTitle("Index", 1);

  histograms.muColl2BxRange = booker.book1D("muBxRangeColl2", (muonColl2Title+" mismatching BX range").c_str(), 5, -2.5, 2.5);
  histograms.muColl2BxRange.setAxisTitle("BX range", 1);
  histograms.muColl2nMu = booker.book1D("nMuColl2", (muonColl2Title+" mismatching muon multiplicity").c_str(), 9, -0.5, 8.5);
  histograms.muColl2nMu.setAxisTitle("Muon multiplicity", 1);
  histograms.muColl2hwPt = booker.book1D("muHwPtColl2", (muonColl2Title+" mismatching muon p_{T}").c_str(), 512, -0.5, 511.5);
  histograms.muColl2hwPt.setAxisTitle("Hardware p_{T}", 1);
  histograms.muColl2hwEta = booker.book1D("muHwEtaColl2", (muonColl2Title+" mismatching muon #eta").c_str(), 461, -230.5, 230.5);
  histograms.muColl2hwEta.setAxisTitle("Hardware #eta", 1);
  histograms.muColl2hwPhi = booker.book1D("muHwPhiColl2", (muonColl2Title+" mismatching muon #phi").c_str(), 576, -0.5, 575.5);
  histograms.muColl2hwPhi.setAxisTitle("Hardware #phi", 1);
  histograms.muColl2hwEtaAtVtx = booker.book1D("muHwEtaAtVtxColl2", (muonColl2Title+" mismatching muon #eta at vertex").c_str(), 461, -230.5, 230.5);
  histograms.muColl2hwEtaAtVtx.setAxisTitle("Hardware #eta at vertex", 1);
  histograms.muColl2hwPhiAtVtx = booker.book1D("muHwPhiAtVtxColl2", (muonColl2Title+" mismatching muon #phi at vertex").c_str(), 576, -0.5, 575.5);
  histograms.muColl2hwPhiAtVtx.setAxisTitle("Hardware #phi at vertex", 1);
  histograms.muColl2hwCharge = booker.book1D("muHwChargeColl2", (muonColl2Title+" mismatching muon charge").c_str(), 2, -0.5, 1.5);
  histograms.muColl2hwCharge.setAxisTitle("Hardware charge", 1);
  histograms.muColl2hwChargeValid = booker.book1D("muHwChargeValidColl2", (muonColl2Title+" mismatching muon charge valid").c_str(), 2, -0.5, 1.5);
  histograms.muColl2hwChargeValid.setAxisTitle("Hardware charge valid", 1);
  histograms.muColl2hwQual = booker.book1D("muHwQualColl2", (muonColl2Title+" mismatching muon quality").c_str(), 16, -0.5, 15.5);
  histograms.muColl2hwQual.setAxisTitle("Hardware quality", 1);
  histograms.muColl2hwIso = booker.book1D("muHwIsoColl2", (muonColl2Title+" mismatching muon isolation").c_str(), 4, -0.5, 3.5);
  histograms.muColl2hwIso.setAxisTitle("Hardware isolation", 1);
  histograms.muColl2Index = booker.book1D("muIndexColl2", (muonColl2Title+" mismatching Input muon index").c_str(), 108, -0.5, 107.5);
  histograms.muColl2Index.setAxisTitle("Index", 1);
}

void L1TStage2MuonComp::dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, muoncompdqm::Histograms const& histograms) const
{

  if (verbose) edm::LogInfo("L1TStage2MuonComp") << "L1TStage2MuonComp: analyze..." << std::endl;

  edm::Handle<l1t::MuonBxCollection> muonBxColl1;
  edm::Handle<l1t::MuonBxCollection> muonBxColl2;
  e.getByToken(muonToken1, muonBxColl1);
  e.getByToken(muonToken2, muonBxColl2);

  histograms.errorSummaryDen.fill(RBXRANGE);
  int bxRange1 = muonBxColl1->getLastBX() - muonBxColl1->getFirstBX() + 1;
  int bxRange2 = muonBxColl2->getLastBX() - muonBxColl2->getFirstBX() + 1;
  if (bxRange1 != bxRange2) {
    histograms.summary.fill(BXRANGEBAD);
    if (incBin[RBXRANGE]) histograms.errorSummaryNum.fill(RBXRANGE);
    int bx;
    for (bx = muonBxColl1->getFirstBX(); bx <= muonBxColl1->getLastBX(); ++bx) {
        histograms.muColl1BxRange.fill(bx);
    }
    for (bx = muonBxColl2->getFirstBX(); bx <= muonBxColl2->getLastBX(); ++bx) {
        histograms.muColl2BxRange.fill(bx);
    }
  } else {
    histograms.summary.fill(BXRANGEGOOD);
  }

  for (int iBx = muonBxColl1->getFirstBX(); iBx <= muonBxColl1->getLastBX(); ++iBx) {
    // don't analyse if this BX does not exist in the second collection
    if (iBx < muonBxColl2->getFirstBX() || iBx > muonBxColl2->getLastBX()) continue;

    l1t::MuonBxCollection::const_iterator muonIt1;
    l1t::MuonBxCollection::const_iterator muonIt2;

    histograms.errorSummaryDen.fill(RNMUON);
    // check number of muons
    if (muonBxColl1->size(iBx) != muonBxColl2->size(iBx)) {
      histograms.summary.fill(NMUONBAD);
      if (incBin[RNMUON]) histograms.errorSummaryNum.fill(RNMUON);
      histograms.muColl1nMu.fill(muonBxColl1->size(iBx));
      histograms.muColl2nMu.fill(muonBxColl2->size(iBx));

      if (muonBxColl1->size(iBx) > muonBxColl2->size(iBx)) {
        muonIt1 = muonBxColl1->begin(iBx) + muonBxColl2->size(iBx);
        for (; muonIt1 != muonBxColl1->end(iBx); ++muonIt1) {
          histograms.muColl1hwPt.fill(muonIt1->hwPt());
          histograms.muColl1hwEta.fill(muonIt1->hwEta());
          histograms.muColl1hwPhi.fill(muonIt1->hwPhi());
          histograms.muColl1hwEtaAtVtx.fill(muonIt1->hwEtaAtVtx());
          histograms.muColl1hwPhiAtVtx.fill(muonIt1->hwPhiAtVtx());
          histograms.muColl1hwCharge.fill(muonIt1->hwCharge());
          histograms.muColl1hwChargeValid.fill(muonIt1->hwChargeValid());
          histograms.muColl1hwQual.fill(muonIt1->hwQual());
          histograms.muColl1hwIso.fill(muonIt1->hwIso());
          histograms.muColl1Index.fill(muonIt1->tfMuonIndex());
        }
      } else {
        muonIt2 = muonBxColl2->begin(iBx) + muonBxColl1->size(iBx);
        for (; muonIt2 != muonBxColl2->end(iBx); ++muonIt2) {
          histograms.muColl2hwPt.fill(muonIt2->hwPt());
          histograms.muColl2hwEta.fill(muonIt2->hwEta());
          histograms.muColl2hwPhi.fill(muonIt2->hwPhi());
          histograms.muColl2hwEtaAtVtx.fill(muonIt2->hwEtaAtVtx());
          histograms.muColl2hwPhiAtVtx.fill(muonIt2->hwPhiAtVtx());
          histograms.muColl2hwCharge.fill(muonIt2->hwCharge());
          histograms.muColl2hwChargeValid.fill(muonIt2->hwChargeValid());
          histograms.muColl2hwQual.fill(muonIt2->hwQual());
          histograms.muColl2hwIso.fill(muonIt2->hwIso());
          histograms.muColl2Index.fill(muonIt2->tfMuonIndex());
        }
      }
    } else {
      histograms.summary.fill(NMUONGOOD);
    }

    muonIt1 = muonBxColl1->begin(iBx);
    muonIt2 = muonBxColl2->begin(iBx);
    while(muonIt1 != muonBxColl1->end(iBx) && muonIt2 != muonBxColl2->end(iBx)) {
      histograms.summary.fill(MUONALL);
      for (int i = RMUON; i <= RIDX; ++i) {
        histograms.errorSummaryDen.fill(i);
      }

      bool muonMismatch = false;    // All muon mismatches
      bool muonSelMismatch = false; // Muon mismatches excluding ignored bins
      if (muonIt1->hwPt() != muonIt2->hwPt()) {
        muonMismatch = true;
        histograms.summary.fill(PTBAD);
        if (incBin[RPT]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RPT);
        }
      }
      if (muonIt1->hwEta() != muonIt2->hwEta()) {
        muonMismatch = true;
        histograms.summary.fill(ETABAD);
        if (incBin[RETA]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RETA);
        }
      }
      if (muonIt1->hwPhi() != muonIt2->hwPhi()) {
        muonMismatch = true;
        histograms.summary.fill(PHIBAD);
        if (incBin[RPHI]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RPHI);
        }
      }
      if (muonIt1->hwEtaAtVtx() != muonIt2->hwEtaAtVtx()) {
        muonMismatch = true;
        histograms.summary.fill(ETAATVTXBAD);
        if (incBin[RETAATVTX]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RETAATVTX);
        }
      }
      if (muonIt1->hwPhiAtVtx() != muonIt2->hwPhiAtVtx()) {
        muonMismatch = true;
        histograms.summary.fill(PHIATVTXBAD);
        if (incBin[RPHIATVTX]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RPHIATVTX);
        }
      }
      if (muonIt1->hwCharge() != muonIt2->hwCharge()) {
        muonMismatch = true;
        histograms.summary.fill(CHARGEBAD);
        if (incBin[RCHARGE]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RCHARGE);
        }
      }
      if (muonIt1->hwChargeValid() != muonIt2->hwChargeValid()) {
        muonMismatch = true;
        histograms.summary.fill(CHARGEVALBAD);
        if (incBin[RCHARGEVAL]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RCHARGEVAL);
        }
      }
      if (muonIt1->hwQual() != muonIt2->hwQual()) {
        muonMismatch = true;
        histograms.summary.fill(QUALBAD);
        if (incBin[RQUAL]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RQUAL);
        }
      }
      if (muonIt1->hwIso() != muonIt2->hwIso()) {
        muonMismatch = true;
        histograms.summary.fill(ISOBAD);
        if (incBin[RISO]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RISO);
        }
      }
      if (muonIt1->tfMuonIndex() != muonIt2->tfMuonIndex()) {
        muonMismatch = true;
        histograms.summary.fill(IDXBAD);
        if (incBin[RIDX]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RIDX);
        }
      }

      if (incBin[RMUON] && muonSelMismatch) {
        histograms.errorSummaryNum.fill(RMUON);
      }

      if (muonMismatch) {

        histograms.muColl1hwPt.fill(muonIt1->hwPt());
        histograms.muColl1hwEta.fill(muonIt1->hwEta());
        histograms.muColl1hwPhi.fill(muonIt1->hwPhi());
        histograms.muColl1hwEtaAtVtx.fill(muonIt1->hwEtaAtVtx());
        histograms.muColl1hwPhiAtVtx.fill(muonIt1->hwPhiAtVtx());
        histograms.muColl1hwCharge.fill(muonIt1->hwCharge());
        histograms.muColl1hwChargeValid.fill(muonIt1->hwChargeValid());
        histograms.muColl1hwQual.fill(muonIt1->hwQual());
        histograms.muColl1hwIso.fill(muonIt1->hwIso());
        histograms.muColl1Index.fill(muonIt1->tfMuonIndex());

        histograms.muColl2hwPt.fill(muonIt2->hwPt());
        histograms.muColl2hwEta.fill(muonIt2->hwEta());
        histograms.muColl2hwPhi.fill(muonIt2->hwPhi());
        histograms.muColl2hwEtaAtVtx.fill(muonIt2->hwEtaAtVtx());
        histograms.muColl2hwPhiAtVtx.fill(muonIt2->hwPhiAtVtx());
        histograms.muColl2hwCharge.fill(muonIt2->hwCharge());
        histograms.muColl2hwChargeValid.fill(muonIt2->hwChargeValid());
        histograms.muColl2hwQual.fill(muonIt2->hwQual());
        histograms.muColl2hwIso.fill(muonIt2->hwIso());
        histograms.muColl2Index.fill(muonIt2->tfMuonIndex());
      } else {
        histograms.summary.fill(MUONGOOD);
      }

      ++muonIt1;
      ++muonIt2;
    }
  }
}

