#include "DQM/L1TMonitor/interface/L1TStage2RegionalMuonCandComp.h"


L1TStage2RegionalMuonCandComp::L1TStage2RegionalMuonCandComp(const edm::ParameterSet& ps)
    : muonToken1(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("regionalMuonCollection1"))),
      muonToken2(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("regionalMuonCollection2"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir")),
      muonColl1Title(ps.getUntrackedParameter<std::string>("regionalMuonCollection1Title")),
      muonColl2Title(ps.getUntrackedParameter<std::string>("regionalMuonCollection2Title")),
      summaryTitle(ps.getUntrackedParameter<std::string>("summaryTitle")),
      ignoreBadTrkAddr(ps.getUntrackedParameter<bool>("ignoreBadTrackAddress")),
      ignoreBin(ps.getUntrackedParameter<std::vector<int>>("ignoreBin")),
      verbose(ps.getUntrackedParameter<bool>("verbose"))
{
  // First include all bins
  for (unsigned int i = 1; i <= RTRACKADDR; i++) {
    incBin[i] = true;
  }
  // Then check the list of bins to ignore
  for (const auto& i : ignoreBin) {
    if (i > 0 && i <= RTRACKADDR) {
      incBin[i] = false;
    }
  }
}

L1TStage2RegionalMuonCandComp::~L1TStage2RegionalMuonCandComp() {}

void L1TStage2RegionalMuonCandComp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("regionalMuonCollection1")->setComment("L1T RegionalMuonCand collection 1");
  desc.add<edm::InputTag>("regionalMuonCollection2")->setComment("L1T RegionalMuonCand collection 2");
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<std::string>("regionalMuonCollection1Title", "Regional muon collection 1")->setComment("Histogram title for first collection.");
  desc.addUntracked<std::string>("regionalMuonCollection2Title", "Regional muon collection 2")->setComment("Histogram title for second collection.");
  desc.addUntracked<std::string>("summaryTitle", "Summary")->setComment("Title of summary histogram.");
  desc.addUntracked<bool>("ignoreBadTrackAddress", false)->setComment("Ignore muon track address mismatches.");
  desc.addUntracked<std::vector<int>>("ignoreBin", std::vector<int>())->setComment("List of bins to ignore");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("l1tStage2RegionalMuonCandComp", desc);
}

void L1TStage2RegionalMuonCandComp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c, regionalmuoncandcompdqm::Histograms& histograms) const
{}

void L1TStage2RegionalMuonCandComp::bookHistograms(DQMStore::ConcurrentBooker& booker, const edm::Run&, const edm::EventSetup&, regionalmuoncandcompdqm::Histograms& histograms) const
{

  std::string trkAddrIgnoreText = "";
  if (ignoreBadTrkAddr) {
    trkAddrIgnoreText = " (Bad track addresses ignored)";
  }

  // Subsystem Monitoring and Muon Output
  booker.setCurrentFolder(monitorDir);

  histograms.summary = booker.book1D("summary", (summaryTitle+trkAddrIgnoreText).c_str(), 17, 1, 18); // range to match bin numbering
  histograms.summary.setBinLabel(BXRANGEGOOD, "BX range match", 1);
  histograms.summary.setBinLabel(BXRANGEBAD, "BX range mismatch", 1);
  histograms.summary.setBinLabel(NMUONGOOD, "muon collection size match", 1);
  histograms.summary.setBinLabel(NMUONBAD, "muon collection size mismatch", 1);
  histograms.summary.setBinLabel(MUONALL, "# muons", 1);
  histograms.summary.setBinLabel(MUONGOOD, "# matching muons", 1);
  histograms.summary.setBinLabel(PTBAD, "p_{T} mismatch", 1);
  histograms.summary.setBinLabel(ETABAD, "#eta mismatch", 1);
  histograms.summary.setBinLabel(LOCALPHIBAD, "local #phi mismatch", 1);
  histograms.summary.setBinLabel(SIGNBAD, "sign mismatch", 1);
  histograms.summary.setBinLabel(SIGNVALBAD, "sign valid mismatch", 1);
  histograms.summary.setBinLabel(QUALBAD, "quality mismatch", 1);
  histograms.summary.setBinLabel(HFBAD, "HF bit mismatch", 1);
  histograms.summary.setBinLabel(LINKBAD, "link mismatch", 1);
  histograms.summary.setBinLabel(PROCBAD, "processor mismatch", 1);
  histograms.summary.setBinLabel(TFBAD, "track finder type mismatch", 1);
  histograms.summary.setBinLabel(TRACKADDRBAD, "track address mismatch", 1);

  histograms.errorSummaryNum = booker.book1D("errorSummaryNum", (summaryTitle+trkAddrIgnoreText).c_str(), 14, 1, 15); // range to match bin numbering
  histograms.errorSummaryNum.setBinLabel(RBXRANGE, "BX range mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RNMUON, "muon collection size mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RMUON, "mismatching muons", 1);
  histograms.errorSummaryNum.setBinLabel(RPT, "p_{T} mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RETA, "#eta mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RLOCALPHI, "local #phi mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RSIGN, "sign mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RSIGNVAL, "sign valid mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RQUAL, "quality mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RHF, "HF bit mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RLINK, "link mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RPROC, "processor mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RTF, "track finder type mismatch", 1);
  histograms.errorSummaryNum.setBinLabel(RTRACKADDR, "track address mismatch", 1);

  // Change the label for those bins that will be ignored
  for (unsigned int i = 1; i <= RTRACKADDR; i++) {
    if (incBin[i]==false) {
      histograms.errorSummaryNum.setBinLabel(i, "Ignored", 1);
    }
  }

  histograms.errorSummaryDen = booker.book1D("errorSummaryDen", "denominators", 14, 1, 15); // range to match bin numbering
  histograms.errorSummaryDen.setBinLabel(RBXRANGE, "# events", 1);
  histograms.errorSummaryDen.setBinLabel(RNMUON, "# muon collections", 1);
  for (int i = RMUON; i <= RTRACKADDR; ++i) {
    histograms.errorSummaryDen.setBinLabel(i, "# muons", 1);
  }

  histograms.muColl1BxRange = booker.book1D("muBxRangeColl1", (muonColl1Title+" mismatching BX range").c_str(), 11, -5.5, 5.5);
  histograms.muColl1BxRange.setAxisTitle("BX range", 1);
  histograms.muColl1nMu = booker.book1D("nMuColl1", (muonColl1Title+" mismatching muon multiplicity").c_str(), 37, -0.5, 36.5);
  histograms.muColl1nMu.setAxisTitle("Muon multiplicity", 1);
  histograms.muColl1hwPt = booker.book1D("muHwPtColl1", (muonColl1Title+" mismatching muon p_{T}"+trkAddrIgnoreText).c_str(), 512, -0.5, 511.5);
  histograms.muColl1hwPt.setAxisTitle("Hardware p_{T}", 1);
  histograms.muColl1hwEta = booker.book1D("muHwEtaColl1", (muonColl1Title+" mismatching muon #eta"+trkAddrIgnoreText).c_str(), 512, -256.5, 255.5);
  histograms.muColl1hwEta.setAxisTitle("Hardware #eta", 1);
  histograms.muColl1hwPhi = booker.book1D("muHwPhiColl1", (muonColl1Title+" mismatching muon #phi"+trkAddrIgnoreText).c_str(), 256, -128.5, 127.5);
  histograms.muColl1hwPhi.setAxisTitle("Hardware #phi", 1);
  histograms.muColl1hwSign = booker.book1D("muHwSignColl1", (muonColl1Title+" mismatching muon sign"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  histograms.muColl1hwSign.setAxisTitle("Hardware sign", 1);
  histograms.muColl1hwSignValid = booker.book1D("muHwSignValidColl1", (muonColl1Title+" mismatching muon sign valid"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  histograms.muColl1hwSignValid.setAxisTitle("Hardware sign valid", 1);
  histograms.muColl1hwQual = booker.book1D("muHwQualColl1", (muonColl1Title+" mismatching muon quality"+trkAddrIgnoreText).c_str(), 16, -0.5, 15.5);
  histograms.muColl1hwQual.setAxisTitle("Hardware quality", 1);
  histograms.muColl1link = booker.book1D("muLinkColl1", (muonColl1Title+" mismatching muon link"+trkAddrIgnoreText).c_str(), 36, 35.5, 71.5);
  histograms.muColl1link.setAxisTitle("Link", 1);
  histograms.muColl1processor = booker.book1D("muProcessorColl1", (muonColl1Title+" mismatching muon processor"+trkAddrIgnoreText).c_str(), 12, -0.5, 15.5);
  histograms.muColl1processor.setAxisTitle("Processor", 1);
  histograms.muColl1trackFinderType = booker.book1D("muTrackFinderTypeColl1", (muonColl1Title+" mismatching muon track finder type"+trkAddrIgnoreText).c_str(), 5, -0.5, 4.5);
  histograms.muColl1trackFinderType.setAxisTitle("Track finder type", 1);
  histograms.muColl1trackFinderType.setBinLabel(BMTFBIN, "BMTF", 1);
  histograms.muColl1trackFinderType.setBinLabel(OMTFNEGBIN, "OMTF-", 1);
  histograms.muColl1trackFinderType.setBinLabel(OMTFPOSBIN, "OMTF+", 1);
  histograms.muColl1trackFinderType.setBinLabel(EMTFNEGBIN, "EMTF-", 1);
  histograms.muColl1trackFinderType.setBinLabel(EMTFPOSBIN, "EMTF+", 1);
  histograms.muColl1hwHF = booker.book1D("muHwHFColl1", (muonColl1Title+" mismatching muon halo/fine-eta bit"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  histograms.muColl1hwHF.setAxisTitle("Hardware H/F bit", 1);
  histograms.muColl1TrkAddrSize = booker.book1D("muTrkAddrSizeColl1", (muonColl1Title+" mismatching muon number of track address keys"+trkAddrIgnoreText).c_str(), 11, -0.5, 10.5);
  histograms.muColl1TrkAddrSize.setAxisTitle("number of keys", 1);
  histograms.muColl1TrkAddr = booker.book2D("muTrkAddrColl1", (muonColl1Title+" mismatching muon track address"+trkAddrIgnoreText).c_str(), 10, -0.5, 9.5, 16, -0.5, 15.5);
  histograms.muColl1TrkAddr.setAxisTitle("key", 1);
  histograms.muColl1TrkAddr.setAxisTitle("value", 2);

  histograms.muColl2BxRange = booker.book1D("muBxRangeColl2", (muonColl2Title+" mismatching BX range").c_str(), 11, -5.5, 5.5);
  histograms.muColl2BxRange.setAxisTitle("BX range", 1);
  histograms.muColl2nMu = booker.book1D("nMuColl2", (muonColl2Title+" mismatching muon multiplicity").c_str(), 37, -0.5, 36.5);
  histograms.muColl2nMu.setAxisTitle("Muon multiplicity", 1);
  histograms.muColl2hwPt = booker.book1D("muHwPtColl2", (muonColl2Title+" mismatching muon p_{T}"+trkAddrIgnoreText).c_str(), 512, -0.5, 511.5);
  histograms.muColl2hwPt.setAxisTitle("Hardware p_{T}", 1);
  histograms.muColl2hwEta = booker.book1D("muHwEtaColl2", (muonColl2Title+" mismatching muon #eta"+trkAddrIgnoreText).c_str(), 512, -256.5, 255.5);
  histograms.muColl2hwEta.setAxisTitle("Hardware #eta", 1);
  histograms.muColl2hwPhi = booker.book1D("muHwPhiColl2", (muonColl2Title+" mismatching muon #phi"+trkAddrIgnoreText).c_str(), 256, -128.5, 127.5);
  histograms.muColl2hwPhi.setAxisTitle("Hardware #phi", 1);
  histograms.muColl2hwSign = booker.book1D("muHwSignColl2", (muonColl2Title+" mismatching muon sign"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  histograms.muColl2hwSign.setAxisTitle("Hardware sign", 1);
  histograms.muColl2hwSignValid = booker.book1D("muHwSignValidColl2", (muonColl2Title+" mismatching muon sign valid"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  histograms.muColl2hwSignValid.setAxisTitle("Hardware sign valid", 1);
  histograms.muColl2hwQual = booker.book1D("muHwQualColl2", (muonColl2Title+" mismatching muon quality"+trkAddrIgnoreText).c_str(), 16, -0.5, 15.5);
  histograms.muColl2hwQual.setAxisTitle("Hardware quality", 1);
  histograms.muColl2link = booker.book1D("muLinkColl2", (muonColl2Title+" mismatching muon link"+trkAddrIgnoreText).c_str(), 36, 35.5, 71.5);
  histograms.muColl2link.setAxisTitle("Link", 1);
  histograms.muColl2processor = booker.book1D("muProcessorColl2", (muonColl2Title+" mismatching muon processor"+trkAddrIgnoreText).c_str(), 12, -0.5, 15.5);
  histograms.muColl2processor.setAxisTitle("Processor", 1);
  histograms.muColl2trackFinderType = booker.book1D("muTrackFinderTypeColl2", (muonColl2Title+" mismatching muon track finder type"+trkAddrIgnoreText).c_str(), 5, -0.5, 4.5);
  histograms.muColl2trackFinderType.setAxisTitle("Track finder type", 1);
  histograms.muColl2trackFinderType.setBinLabel(BMTFBIN, "BMTF", 1);
  histograms.muColl2trackFinderType.setBinLabel(OMTFNEGBIN, "OMTF-", 1);
  histograms.muColl2trackFinderType.setBinLabel(OMTFPOSBIN, "OMTF+", 1);
  histograms.muColl2trackFinderType.setBinLabel(EMTFNEGBIN, "EMTF-", 1);
  histograms.muColl2trackFinderType.setBinLabel(EMTFPOSBIN, "EMTF+", 1);
  histograms.muColl2hwHF = booker.book1D("muHwHFColl2", (muonColl2Title+" mismatching muon halo/fine-eta bit"+trkAddrIgnoreText).c_str(), 2, -0.5, 1.5);
  histograms.muColl2hwHF.setAxisTitle("Hardware H/F bit", 1);
  histograms.muColl2TrkAddrSize = booker.book1D("muTrkAddrSizeColl2", (muonColl2Title+" mismatching muon number of track address keys"+trkAddrIgnoreText).c_str(), 11, -0.5, 10.5);
  histograms.muColl2TrkAddrSize.setAxisTitle("number of keys", 1);
  histograms.muColl2TrkAddr = booker.book2D("muTrkAddrColl2", (muonColl2Title+" mismatching muon track address"+trkAddrIgnoreText).c_str(), 10, -0.5, 9.5, 16, -0.5, 15.5);
  histograms.muColl2TrkAddr.setAxisTitle("key", 1);
  histograms.muColl2TrkAddr.setAxisTitle("value", 2);
}

void L1TStage2RegionalMuonCandComp::dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, regionalmuoncandcompdqm::Histograms const& histograms) const
{

  if (verbose) edm::LogInfo("L1TStage2RegionalMuonCandComp") << "L1TStage2RegionalMuonCandComp: analyze..." << std::endl;

  edm::Handle<l1t::RegionalMuonCandBxCollection> muonBxColl1;
  edm::Handle<l1t::RegionalMuonCandBxCollection> muonBxColl2;
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

    l1t::RegionalMuonCandBxCollection::const_iterator muonIt1;
    l1t::RegionalMuonCandBxCollection::const_iterator muonIt2;

    histograms.errorSummaryDen.fill(RNMUON);
    // check number of muons
    if (muonBxColl1->size(iBx) != muonBxColl2->size(iBx)) {
      histograms.summary.fill(NMUONBAD);
      if (incBin[RNMUON]) histograms.errorSummaryNum.fill(RNMUON);
      histograms.muColl1nMu.fill(muonBxColl1->size(iBx));
      histograms.muColl2nMu.fill(muonBxColl2->size(iBx));

      if (muonBxColl1->size(iBx) > muonBxColl2->size(iBx)) {
        muonIt1 = muonBxColl1->begin(iBx) + muonBxColl2->size(iBx);
        const std::map<int, int> muon1TrackAddr = muonIt1->trackAddress();
        for (; muonIt1 != muonBxColl1->end(iBx); ++muonIt1) {
          histograms.muColl1hwPt.fill(muonIt1->hwPt());
          histograms.muColl1hwEta.fill(muonIt1->hwEta());
          histograms.muColl1hwPhi.fill(muonIt1->hwPhi());
          histograms.muColl1hwSign.fill(muonIt1->hwSign());
          histograms.muColl1hwSignValid.fill(muonIt1->hwSignValid());
          histograms.muColl1hwQual.fill(muonIt1->hwQual());
          histograms.muColl1link.fill(muonIt1->link());
          histograms.muColl1processor.fill(muonIt1->processor());
          histograms.muColl1trackFinderType.fill(muonIt1->trackFinderType());
          histograms.muColl1hwHF.fill(muonIt1->hwHF());
          histograms.muColl1TrkAddrSize.fill(muon1TrackAddr.size());
          for (std::map<int, int>::const_iterator trIt1 = muon1TrackAddr.begin(); trIt1 != muon1TrackAddr.end(); ++trIt1) {
            histograms.muColl1TrkAddr.fill(trIt1->first, trIt1->second);
          }
        }
      } else {
        muonIt2 = muonBxColl2->begin(iBx) + muonBxColl1->size(iBx);
        const std::map<int, int> muon2TrackAddr = muonIt2->trackAddress();
        for (; muonIt2 != muonBxColl2->end(iBx); ++muonIt2) {
          histograms.muColl2hwPt.fill(muonIt2->hwPt());
          histograms.muColl2hwEta.fill(muonIt2->hwEta());
          histograms.muColl2hwPhi.fill(muonIt2->hwPhi());
          histograms.muColl2hwSign.fill(muonIt2->hwSign());
          histograms.muColl2hwSignValid.fill(muonIt2->hwSignValid());
          histograms.muColl2hwQual.fill(muonIt2->hwQual());
          histograms.muColl2link.fill(muonIt2->link());
          histograms.muColl2processor.fill(muonIt2->processor());
          histograms.muColl2trackFinderType.fill(muonIt2->trackFinderType());
          histograms.muColl2hwHF.fill(muonIt2->hwHF());
          histograms.muColl2TrkAddrSize.fill(muon2TrackAddr.size());
          for (std::map<int, int>::const_iterator trIt2 = muon2TrackAddr.begin(); trIt2 != muon2TrackAddr.end(); ++trIt2) {
            histograms.muColl2TrkAddr.fill(trIt2->first, trIt2->second);
          }
        }
      }
    } else {
      histograms.summary.fill(NMUONGOOD);
    }

    muonIt1 = muonBxColl1->begin(iBx);
    muonIt2 = muonBxColl2->begin(iBx);
    //std::cout << "Analysing muons from BX " << iBx << std::endl;
    while(muonIt1 != muonBxColl1->end(iBx) && muonIt2 != muonBxColl2->end(iBx)) {
      //std::cout << "Coll 1 muon: hwPt=" << muonIt1->hwPt() << ", hwEta=" << muonIt1->hwEta() << ", hwPhi=" << muonIt1->hwPhi()
      //          << ", hwSign=" << muonIt1->hwSign() << ", hwSignValid=" << muonIt1->hwSignValid()
      //          << ", hwQual=" << muonIt1->hwQual() << ", link=" << muonIt1->link() << ", processor=" << muonIt1->processor()
      //          << ", trackFinderType=" << muonIt1->trackFinderType() << std::endl;
      //std::cout << "Coll 2 muon: hwPt=" << muonIt2->hwPt() << ", hwEta=" << muonIt2->hwEta() << ", hwPhi=" << muonIt2->hwPhi()
      //          << ", hwSign=" << muonIt2->hwSign() << ", hwSignValid=" << muonIt2->hwSignValid()
      //          << ", hwQual=" << muonIt2->hwQual() << ", link=" << muonIt2->link() << ", processor=" << muonIt2->processor()
      //          << ", trackFinderType=" << muonIt2->trackFinderType() << std::endl;
      histograms.summary.fill(MUONALL);
      for (int i = RMUON; i <= RTRACKADDR; ++i) {
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
        histograms.summary.fill(LOCALPHIBAD);
        if (incBin[RLOCALPHI]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RLOCALPHI);
        }
      }
      if (muonIt1->hwSign() != muonIt2->hwSign()) {
        muonMismatch = true;
        histograms.summary.fill(SIGNBAD);
        if (incBin[RSIGN]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RSIGN);
        }
      }
      if (muonIt1->hwSignValid() != muonIt2->hwSignValid()) {
        muonMismatch = true;
        histograms.summary.fill(SIGNVALBAD);
        if (incBin[RSIGNVAL]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RSIGNVAL);
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
      if (muonIt1->hwHF() != muonIt2->hwHF()) {
        muonMismatch = true;
        histograms.summary.fill(HFBAD);
        if (incBin[RHF]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RHF);
        }
      }
      if (muonIt1->link() != muonIt2->link()) {
        muonMismatch = true;
        histograms.summary.fill(LINKBAD);
        if (incBin[RLINK]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RLINK);
        }
      }
      if (muonIt1->processor() != muonIt2->processor()) {
        muonMismatch = true;
        histograms.summary.fill(PROCBAD);
        if (incBin[RPROC]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RPROC);
        }
      }
      if (muonIt1->trackFinderType() != muonIt2->trackFinderType()) {
        muonMismatch = true;
        histograms.summary.fill(TFBAD);
        if (incBin[RTF]) {
          muonSelMismatch = true;
          histograms.errorSummaryNum.fill(RTF);
        }
      }
      // check track address
      const std::map<int, int> muon1TrackAddr = muonIt1->trackAddress();
      std::map<int, int> muon2TrackAddr = muonIt2->trackAddress();
      bool badTrackAddr = false;
      if (muon1TrackAddr.size() == muon2TrackAddr.size()) {
        for (std::map<int, int>::const_iterator trIt1 = muon1TrackAddr.begin(); trIt1 != muon1TrackAddr.end(); ++trIt1) {
          if (muon2TrackAddr.find(trIt1->first) == muon2TrackAddr.end()) { // key does not exist
            badTrackAddr = true;
            break;
          } else if (muon2TrackAddr[trIt1->first] != trIt1->second) { // wrong value for key
            badTrackAddr = true;
            break;
          }
        }
      } else {
        badTrackAddr = true;
      }
      if (badTrackAddr) {
        if (!ignoreBadTrkAddr) {
          muonMismatch = true;
          if (incBin[RTRACKADDR]) muonSelMismatch = true;
        }
        histograms.summary.fill(TRACKADDRBAD);
        if (incBin[RTRACKADDR]) histograms.errorSummaryNum.fill(RTRACKADDR);
      }

      if (incBin[RMUON] && muonSelMismatch) {
        histograms.errorSummaryNum.fill(RMUON);
      }

      if (muonMismatch) {

        histograms.muColl1hwPt.fill(muonIt1->hwPt());
        histograms.muColl1hwEta.fill(muonIt1->hwEta());
        histograms.muColl1hwPhi.fill(muonIt1->hwPhi());
        histograms.muColl1hwSign.fill(muonIt1->hwSign());
        histograms.muColl1hwSignValid.fill(muonIt1->hwSignValid());
        histograms.muColl1hwQual.fill(muonIt1->hwQual());
        histograms.muColl1link.fill(muonIt1->link());
        histograms.muColl1processor.fill(muonIt1->processor());
        histograms.muColl1trackFinderType.fill(muonIt1->trackFinderType());
        histograms.muColl1hwHF.fill(muonIt1->hwHF());
        histograms.muColl1TrkAddrSize.fill(muon1TrackAddr.size());
        for (std::map<int, int>::const_iterator trIt1 = muon1TrackAddr.begin(); trIt1 != muon1TrackAddr.end(); ++trIt1) {
          histograms.muColl1TrkAddr.fill(trIt1->first, trIt1->second);
        }

        histograms.muColl2hwPt.fill(muonIt2->hwPt());
        histograms.muColl2hwEta.fill(muonIt2->hwEta());
        histograms.muColl2hwPhi.fill(muonIt2->hwPhi());
        histograms.muColl2hwSign.fill(muonIt2->hwSign());
        histograms.muColl2hwSignValid.fill(muonIt2->hwSignValid());
        histograms.muColl2hwQual.fill(muonIt2->hwQual());
        histograms.muColl2link.fill(muonIt2->link());
        histograms.muColl2processor.fill(muonIt2->processor());
        histograms.muColl2trackFinderType.fill(muonIt2->trackFinderType());
        histograms.muColl2hwHF.fill(muonIt2->hwHF());
        histograms.muColl2TrkAddrSize.fill(muon2TrackAddr.size());
        for (std::map<int, int>::const_iterator trIt2 = muon2TrackAddr.begin(); trIt2 != muon2TrackAddr.end(); ++trIt2) {
          histograms.muColl2TrkAddr.fill(trIt2->first, trIt2->second);
        }
      } else {
        histograms.summary.fill(MUONGOOD);
      }

      ++muonIt1;
      ++muonIt2;
    }
  }
}

