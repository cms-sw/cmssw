#include "DQM/L1TMonitor/interface/L1TdeStage2CaloLayer2.h"

L1TdeStage2CaloLayer2::L1TdeStage2CaloLayer2 (const edm::ParameterSet& ps)
  : monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
    calol2JetCollectionData(consumes <l1t::JetBxCollection>(
                              ps.getParameter<edm::InputTag>(
                                "calol2JetCollectionData"))),
    calol2JetCollectionEmul(consumes <l1t::JetBxCollection>(
                              ps.getParameter<edm::InputTag>(
                                "calol2JetCollectionEmul"))),
    calol2EGammaCollectionData(consumes <l1t::EGammaBxCollection>(
                                 ps.getParameter<edm::InputTag>(
                                   "calol2EGammaCollectionData"))),
    calol2EGammaCollectionEmul(consumes <l1t::EGammaBxCollection>(
                                 ps.getParameter<edm::InputTag>(
                                   "calol2EGammaCollectionEmul"))),
    calol2TauCollectionData(consumes <l1t::TauBxCollection>(
                              ps.getParameter<edm::InputTag>(
                                "calol2TauCollectionData"))),
    calol2TauCollectionEmul(consumes <l1t::TauBxCollection>(
                              ps.getParameter<edm::InputTag>(
                                "calol2TauCollectionEmul"))),
    calol2EtSumCollectionData(consumes <l1t::EtSumBxCollection>(
                                ps.getParameter<edm::InputTag>(
                                  "calol2EtSumCollectionData"))),
    calol2EtSumCollectionEmul(consumes <l1t::EtSumBxCollection>(
                                ps.getParameter<edm::InputTag>(
                                  "calol2EtSumCollectionEmul"))),
    verbose(ps.getUntrackedParameter<bool> ("verbose", false))
{}

void L1TdeStage2CaloLayer2::bookHistograms(
  DQMStore::ConcurrentBooker &booker,
  edm::Run const &,
  edm::EventSetup const&,
  calolayer2dedqm::Histograms &histograms) const {

  // DQM directory to store histograms with problematic jets
  booker.setCurrentFolder(monitorDir + "/Problematic Jets candidates");

  histograms.jetEtData = booker.book1D("Problematic Data Jet iEt", "Jet iE_{T}",
                             1400, 0, 1400);
  histograms.jetEtaData = booker.book1D("Problematic Data Jet iEta", "Jet i#eta",
                              227, -113.5, 113.5);
  histograms.jetPhiData = booker.book1D("Problematic Data Jet iPhi", "Jet i#phi",
                              288, -0.5, 143.5);
  histograms.jetEtEmul = booker.book1D("Problematic Emul Jet iEt", "Jet iE_{T}",
                             1400, 0, 1400);
  histograms.jetEtaEmul = booker.book1D("Problematic Emul Jet iEta", "Jet i#eta",
                              227, -113.5, 113.5);
  histograms.jetPhiEmul = booker.book1D("Problematic Emul Jet iPhi", "Jet i#phi",
                              288, -0.5, 143.5);

  // DQM directory to store histograms with problematic e/gs
  booker.setCurrentFolder(monitorDir + "/Problematic EG candidtes");

  histograms.egEtData = booker.book1D("Problematic Data Eg iEt", "Eg iE_{T}",
                            1400, 0, 1400);
  histograms.egEtaData = booker.book1D("Problematic Data Eg iEta", "Eg i#eta",
                             227, -113.5, 113.5);
  histograms.egPhiData = booker.book1D("Problematic Data Eg iPhi", "Eg i#phi",
                             288, -0.5, 143.5);
  histograms.egEtEmul = booker.book1D("Problematic Emul Eg iEt", "Eg iE_{T}",
                            1400, 0, 1400);
  histograms.egEtaEmul = booker.book1D("Problematic Emul Eg iEta", "Eg i#eta",
                             227, -113.5, 113.5);
  histograms.egPhiEmul = booker.book1D("Problematic Emul Eg iPhi", "Eg i#phi",
                             288, -0.5, 143.5);

  histograms.isoEgEtData = booker.book1D("Problematic Isolated Data Eg iEt",
                               "Iso Eg iE_{T}", 1400, 0, 1400);
  histograms.isoEgEtaData = booker.book1D("Problematic Isolated Data Eg iEta",
                                "Iso Eg i#eta", 227, -113.5, 113.5);
  histograms.isoEgPhiData = booker.book1D("Problematic Isolated Data Eg iPhi",
                                "Iso Eg i#phi", 288, -0.5, 143.5);
  histograms.isoEgEtEmul = booker.book1D("Problematic Isolated Emul Eg iEt",
                               "Iso Eg iE_{T}", 1400, 0, 1400);
  histograms.isoEgEtaEmul = booker.book1D("Problematic Isolated Emul Eg iEta",
                                "Iso Eg i#eta", 227, -113.5, 113.5);
  histograms.isoEgPhiEmul = booker.book1D("Problematic Isolated Emul Eg iPhi",
                                "Iso Eg i#phi", 288, -0.5, 143.5);

  // DQM directory to store histograms with problematic taus
  booker.setCurrentFolder(monitorDir + "/Problematic Tau candidtes");

  histograms.tauEtData = booker.book1D("Problematic Data Tau iEt", "Tau iE_{T}",
                             1400, 0, 1400);
  histograms.tauEtaData = booker.book1D("Problematic Data Tau iEta", "Tau i#eta",
                              227, -113.5, 113.5);
  histograms.tauPhiData = booker.book1D("Problematic Data Tau iPhi", "Tau i#phi",
                              288, -0.5, 143.5);
  histograms.tauEtEmul = booker.book1D("Problematic Emul Tau iEt", "Tau iE_{T}",
                             1400, 0, 1400);
  histograms.tauEtaEmul = booker.book1D("Problematic Emul Tau iEta", "Tau i#eta",
                              227, -113.5, 113.5);
  histograms.tauPhiEmul = booker.book1D("Problematic Emul Tau iPhi", "Tau i#phi",
                              288, -0.5, 143.5);

  histograms.isoTauEtData = booker.book1D("Problematic Isolated Data Tau iEt",
                                "Iso Tau iE_{T}", 1400, 0, 1400);
  histograms.isoTauEtaData = booker.book1D("Problematic Isolated Data Tau iEta",
                                 "Iso Tau i#eta", 227, -113.5, 113.5);
  histograms.isoTauPhiData = booker.book1D("Problematic Isolated Data Tau iPhi",
                                 "Iso Tau i#phi", 288, -0.5, 143.5);
  histograms.isoTauEtEmul = booker.book1D("Problematic Isolated Emul Tau iEt",
                                "Iso Tau iE_{T}", 1400, 0, 1400);
  histograms.isoTauEtaEmul = booker.book1D("Problematic Isolated Emul Tau iEta",
                                 "Iso Tau i#eta", 227, -113.5, 113.5);
  histograms.isoTauPhiEmul = booker.book1D("Problematic Isolated Emul Tau iPhi",
                                 "Iso Tau i#phi", 288, -0.5, 143.5);

  // DQM directory to store histograms with problematic sums
  booker.setCurrentFolder(monitorDir + "/Problematic Sums");

  // book ETT type sums
  histograms.ettData = booker.book1D("Problematic ETT Sum - Data", "ETT iE_{T}",
                           7000, -0.5, 6999.5);
  histograms.ettEmul = booker.book1D("Problematic ETT Sum - Emulator", "ETT iE_{T}",
                           7000, -0.5, 6999.5);
  histograms.ettHFData = booker.book1D("Problematic ETTHF Sum - Data", "ETTHF iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.ettHFEmul = booker.book1D("Problematic ETTHF Sum - Emulator", "ETTHF iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.ettEmData = booker.book1D("Problematic ETTEM Sum - Data", "ETTEM iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.ettEmEmul = booker.book1D("Problematic ETTEM Sum - Emulator", "ETTEM iE_{T}",
                             7000, -0.5, 6999.5);

  // book HTT type sums
  histograms.httData = booker.book1D("Problematic HTT Sum - Data", "HTT iE_{T}",
                           7000, -0.5, 6999.5);
  histograms.httEmul = booker.book1D("Problematic HTT Sum - Emulator", "HTT iE_{T}",
                           7000, -0.5, 6999.5);
  histograms.httHFData = booker.book1D("Problematic HTTHF Sum - Data", "HTTHF iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.httHFEmul = booker.book1D("Problematic HTTHF Sum - Emulator", "HTTHF iE_{T}",
                             7000, -0.5, 6999.5);

  // book MET type sums
  histograms.metEtData = booker.book1D("Problematic MET Sum Et - Data", "MET iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.metEtEmul = booker.book1D("Problematic MET Sum Et - Emulator", "MET iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.metPhiData = booker.book1D("Problematic MET Sum phi - Data", "MET i#phi",
                              1008, -0.5, 1007.5);
  histograms.metPhiEmul = booker.book1D("Problematic MET Sum phi - Emulator", "MET i#phi",
                              1008, -0.5, 1007.5);

  histograms.metHFEtData = booker.book1D("Problematic METHF Sum Et - Data",
                               "METHF iE_{T}", 7000, -0.5, 6999.5);
  histograms.metHFEtEmul = booker.book1D("Problematic METHF Sum Et - Emulator",
                               "METHF iE_{T}", 7000, -0.5, 6999.5);
  histograms.metHFPhiData = booker.book1D("Problematic METHF Sum phi - Data",
                                "METHF i#phi", 1008, -0.5, 1007.5);
  histograms.metHFPhiEmul = booker.book1D("Problematic METHF Sum phi - Emulator",
                                "METHF i#phi", 1008, -0.5, 1007.5);

  // book MHT type sums
  histograms.mhtEtData = booker.book1D("Problematic MHT Sum Et - Data", "MHT iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.mhtEtEmul = booker.book1D("Problematic MHT Sum Et - Emulator", "MHT iE_{T}",
                             7000, -0.5, 6999.5);
  histograms.mhtPhiData = booker.book1D("Problematic MHT Sum phi - Data", "MHT i#phi",
                              1008, -0.5, 1007.5);
  histograms.mhtPhiEmul = booker.book1D("Problematic MHT Sum phi - Emulator", "MHT i#phi",
                              1008, -0.5, 1007.5);

  histograms.mhtHFEtData = booker.book1D("Problematic MHTHF Sum Et - Data",
                               "MHTHF iE_{T}", 7000, -0.5, 6999.5);
  histograms.mhtHFEtEmul = booker.book1D("Problematic MHTHF Sum Et - Emulator",
                               "MHTHF iE_{T}", 7000, -0.5, 6999.5);
  histograms.mhtHFPhiData = booker.book1D("Problematic MHTHF Sum phi - Data",
                                "MHTHF i#phi", 1008, -0.5, 1007.5);
  histograms.mhtHFPhiEmul = booker.book1D("Problematic MHTHF Sum phi - Emulator",
                                "MHTHF i#phi", 1008, -0.5, 1007.5);

  // book minimum bias sums
  histograms.mbhfp0Data = booker.book1D("Problematic MBHFP0 Sum - Data",
                              "", 16, -0.5, 15.5);
  histograms.mbhfp0Emul = booker.book1D("Problematic MBHFP0 Sum - Emulator",
                              "", 16, -0.5, 15.5);
  histograms.mbhfm0Data = booker.book1D("Problematic MBHFM0 Sum - Data",
                              "", 16, -0.5, 15.5);
  histograms.mbhfm0Emul = booker.book1D("Problematic MBHFM0 Sum - Emulator",
                              "", 16, -0.5, 15.5);
  histograms.mbhfm1Data = booker.book1D("Problematic MBHFM1 Sum - Data",
                              "", 16, -0.5, 15.5);
  histograms.mbhfm1Emul = booker.book1D("Problematic MBHFM1 Sum - Emulator",
                              "", 16, -0.5, 15.5);
  histograms.mbhfp1Data = booker.book1D("Problematic MBHFP1 Sum - Data",
                              "", 16, -0.5, 15.5);
  histograms.mbhfp1Emul = booker.book1D("Problematic MBHFP1 Sum - Emulator",
                              "", 16, -0.5, 15.5);

  // book tower count sums
  histograms.towCountData = booker.book1D("Problematic Tower Count Sum - Data",
                                "", 5904, -0.5, 5903.5);
  histograms.towCountEmul = booker.book1D("Problematic Tower Count Sum - Emulator",
                                "", 5904, -0.5, 5903.5);
  // for reference on arguments of book2D, see
  // https://cmssdt.cern.ch/SDT/doxygen/CMSSW_8_0_24/doc/html/df/d26/DQMStore_8cc_source.html#l01070


  // setup the directory where the histograms are to be visualised, value is set
  // in constructor and taken from python configuration file for module
  booker.setCurrentFolder(monitorDir + "/expert");


  // Jet energy in MP firmware is stored in 16 bits which sets the range of
  // jet energy to 2^16 * 0.5 GeV = 32768 GeV (65536 hardware units)
  // --- this is only for MP jets, the demux jets have much decreased precision
  // --- and this should be replaced

  // the index of the first bin in histogram should match value of first enum
  histograms.agreementSummary = booker.book1D(
    "CaloL2 Object Agreement Summary",
    "CaloL2 event-by-event object agreement fractions", 10, 1, 11);

  histograms.agreementSummary.setBinLabel(EVENTGOOD, "good events");
  histograms.agreementSummary.setBinLabel(NEVENTS, "total events");
  histograms.agreementSummary.setBinLabel(NJETS_S, "total jets");
  histograms.agreementSummary.setBinLabel(JETGOOD_S, "good jets");
  histograms.agreementSummary.setBinLabel(NEGS_S, "total e/gs");
  histograms.agreementSummary.setBinLabel(EGGOOD_S, "good e/gs");
  histograms.agreementSummary.setBinLabel(NTAUS_S, "total taus");
  histograms.agreementSummary.setBinLabel(TAUGOOD_S, "good taus");
  histograms.agreementSummary.setBinLabel(NSUMS_S, "total sums");
  histograms.agreementSummary.setBinLabel(SUMGOOD_S, "good sums");

  histograms.jetSummary = booker.book1D(
    "Jet Agreement Summary", "Jet Agreement Summary", 4, 1, 5);
  histograms.jetSummary.setBinLabel(NJETS, "total jets");
  histograms.jetSummary.setBinLabel(JETGOOD, "good jets");
  histograms.jetSummary.setBinLabel(JETPOSOFF, "jets pos off only");
  histograms.jetSummary.setBinLabel(JETETOFF, "jets Et off only ");

  histograms.egSummary = booker.book1D(
    "EG Agreement Summary", "EG Agreement Summary", 8, 1, 9);
  histograms.egSummary.setBinLabel(NEGS, "total non-iso e/gs");
  histograms.egSummary.setBinLabel(EGGOOD, "good non-iso e/gs");
  histograms.egSummary.setBinLabel(EGPOSOFF, "non-iso e/gs pos off");
  histograms.egSummary.setBinLabel(EGETOFF, "non-iso e/gs Et off");
  histograms.egSummary.setBinLabel(NISOEGS, "total iso e/gs");
  histograms.egSummary.setBinLabel(ISOEGGOOD, "good iso e/gs");
  histograms.egSummary.setBinLabel(ISOEGPOSOFF, "iso e/gs pos off");
  histograms.egSummary.setBinLabel(ISOEGETOFF, "iso e/gs Et off");

  histograms.tauSummary = booker.book1D(
    "Tau Agreement Summary", "Tau Agreement Summary", 8, 1, 9);
  histograms.tauSummary.setBinLabel(NTAUS, "total taus");
  histograms.tauSummary.setBinLabel(TAUGOOD, "good non-iso taus");
  histograms.tauSummary.setBinLabel(TAUPOSOFF, "non-iso taus pos off");
  histograms.tauSummary.setBinLabel(TAUETOFF, "non-iso taus Et off");
  histograms.tauSummary.setBinLabel(NISOTAUS, "total iso taus");
  histograms.tauSummary.setBinLabel(ISOTAUGOOD, "good iso taus");
  histograms.tauSummary.setBinLabel(ISOTAUPOSOFF, "iso taus pos off");
  histograms.tauSummary.setBinLabel(ISOTAUETOFF, "iso taus Et off");

  histograms.sumSummary = booker.book1D(
    "Energy Sum Agreement Summary", "Sum Agreement Summary", 14, 1, 15);
  histograms.sumSummary.setBinLabel(NSUMS, "total sums");
  histograms.sumSummary.setBinLabel(SUMGOOD, "good sums");
  histograms.sumSummary.setBinLabel(NETTSUMS, "total ETT sums");
  histograms.sumSummary.setBinLabel(ETTSUMGOOD, "good ETT sums");
  histograms.sumSummary.setBinLabel(NHTTSUMS, "total HTT sums");
  histograms.sumSummary.setBinLabel(HTTSUMGOOD, "good HTT sums");
  histograms.sumSummary.setBinLabel(NMETSUMS, "total MET sums");
  histograms.sumSummary.setBinLabel(METSUMGOOD, "good MET sums");
  histograms.sumSummary.setBinLabel(NMHTSUMS, "total MHT sums");
  histograms.sumSummary.setBinLabel(MHTSUMGOOD, "good MHT sums");
  histograms.sumSummary.setBinLabel(NMBHFSUMS, "total MBHF sums");
  histograms.sumSummary.setBinLabel(MBHFSUMGOOD, "good MBHF sums");
  histograms.sumSummary.setBinLabel(NTOWCOUNTS, "total TowCount sums");
  histograms.sumSummary.setBinLabel(TOWCOUNTGOOD, "good TowCount sums");

  // high level directory
  booker.setCurrentFolder(monitorDir);

  histograms.problemSummary = booker.book1D(
    "Problem Summary", "Problematic Event Summary", 8, 1, 9);
  histograms.problemSummary.setBinLabel(NEVENTS_P, "total events");
  histograms.problemSummary.setBinLabel(JETCOLLSIZE, "jet collection size");
  histograms.problemSummary.setBinLabel(EGCOLLSIZE, "eg collection size");
  histograms.problemSummary.setBinLabel(TAUCOLLSIZE, "tau collection size");
  histograms.problemSummary.setBinLabel(JETMISMATCH, "jet mismatch");
  histograms.problemSummary.setBinLabel(EGMISMATCH, "eg mismatch");
  histograms.problemSummary.setBinLabel(TAUMISMATCH, "tau mismatch");
  histograms.problemSummary.setBinLabel(SUMMISMATCH, "sum mismatch");
}
void L1TdeStage2CaloLayer2::dqmAnalyze (
  const edm::Event& e,
  const edm::EventSetup & c,
  const calolayer2dedqm::Histograms& histograms) const {

  if (verbose)
    edm::LogInfo("L1TdeStage2CaloLayer2") << "L1TdeStage2CaloLayer2: analyse "
                                          << std::endl;

  // define collections to hold lists of objects in event
  edm::Handle<l1t::JetBxCollection> jetDataCol;
  edm::Handle<l1t::JetBxCollection> jetEmulCol;
  edm::Handle<l1t::EGammaBxCollection> egDataCol;
  edm::Handle<l1t::EGammaBxCollection> egEmulCol;
  edm::Handle<l1t::TauBxCollection> tauDataCol;
  edm::Handle<l1t::TauBxCollection> tauEmulCol;
  edm::Handle<l1t::EtSumBxCollection> sumDataCol;
  edm::Handle<l1t::EtSumBxCollection> sumEmulCol;

  // map event contents to above collections
  e.getByToken(calol2JetCollectionData, jetDataCol);
  e.getByToken(calol2JetCollectionEmul, jetEmulCol);
  e.getByToken(calol2EGammaCollectionData, egDataCol);
  e.getByToken(calol2EGammaCollectionEmul, egEmulCol);
  e.getByToken(calol2TauCollectionData, tauDataCol);
  e.getByToken(calol2TauCollectionEmul, tauEmulCol);
  e.getByToken(calol2EtSumCollectionData, sumDataCol);
  e.getByToken(calol2EtSumCollectionEmul, sumEmulCol);

  bool eventGood = true;

  // we assume that the first and last bx of the emulator data is 0 since it is
  // very unlikely to have received RAW data from more than just the triggered
  // bx

  /**
     Notes:
     - The hardware can send up to 12 jets due to bandwidth limitation so it
       will sort the jets it has found in order of decreasing pT and will send
       only the top 12. The emulator does not have similar constraint but can be
       configured to truncate the list of jets it has found. In the case that a
       small number of jets is found (less than 12), the full list will be sent.
     - Currently, the edge case where the number of jets/objects in data and
       emulator are different is being skipped but would need to be addressed
       before the module can be declared complete.

     Edge cases to consider:
     - there are more emulator jets than data jets
     - there are more data jets than emulator jets
     - missing jet is at beginning/end
     - missing jet is in the middle
  */

  if (!compareJets(jetDataCol, jetEmulCol, histograms)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: jet problem "
                                            << std::endl;
    histograms.problemSummary.fill(JETMISMATCH);
    eventGood = false;
  }

  if (!compareEGs(egDataCol, egEmulCol, histograms)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: eg problem "
                                            << std::endl;
    histograms.problemSummary.fill(EGMISMATCH);
    eventGood = false;
  }

  if (!compareTaus(tauDataCol, tauEmulCol, histograms)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: tau problem "
                                            << std::endl;
    histograms.problemSummary.fill(TAUMISMATCH);
    eventGood = false;
  }

  if (!compareSums(sumDataCol, sumEmulCol, histograms)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: sum problem "
                                            << std::endl;
    histograms.problemSummary.fill(SUMMISMATCH);
    eventGood = false;
  }

  /**
     Questions:
     - what could make the data and emul bx ranges to be different?
     - how can I confirm that the emulator data is being filled?
  */

  if (eventGood) {
    histograms.agreementSummary.fill(EVENTGOOD);
  }

  histograms.agreementSummary.fill(NEVENTS);
  histograms.problemSummary.fill(NEVENTS_P);
}

// comparison method for jets
bool L1TdeStage2CaloLayer2::compareJets(
  const edm::Handle<l1t::JetBxCollection> & dataCol,
  const edm::Handle<l1t::JetBxCollection> & emulCol,
  const calolayer2dedqm::Histograms & histograms) const
{
  bool eventGood = true;

  l1t::JetBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::JetBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  // process jets
  if (dataCol->size(currBx) != emulCol->size(currBx)) {

    if (dataCol->size(currBx) < emulCol->size(currBx)) {

      // if only the data collection is empty, declare event as bad
      if (dataCol->isEmpty(currBx)) return false;

      while (true) {
        histograms.jetEtData.fill(dataIt->hwPt());
        histograms.jetEtaData.fill(dataIt->hwEta());
        histograms.jetPhiData.fill(dataIt->hwPhi());

        ++dataIt;

        if (dataIt == dataCol->end(currBx))
          break;
      }
    } else {

      // if only the emul collection is empty, declare event as bad
      if (emulCol->isEmpty(currBx)) return false;

      while (true) {

        histograms.jetEtEmul.fill(emulIt->hwPt());
        histograms.jetEtaEmul.fill(emulIt->hwEta());
        histograms.jetPhiEmul.fill(emulIt->hwPhi());

        ++emulIt;

        if (emulIt == emulCol->end(currBx))
          break;
      }

      while (true) {

        histograms.jetEtEmul.fill(dataIt->hwPt());
        histograms.jetEtaEmul.fill(dataIt->hwEta());
        histograms.jetPhiEmul.fill(dataIt->hwPhi());

        ++dataIt;

        if (dataIt == dataCol->end(currBx))
          break;
      }
    }

    histograms.problemSummary.fill(JETCOLLSIZE);
    return false;
  }

  int nJets = 0;
  if (dataIt != dataCol->end(currBx) ||
      emulIt != emulCol->end(currBx)) {
    while(true) {

      ++nJets;

      bool posGood = true;
      bool etGood = true;

      // object pt mismatch
      if (dataIt->hwPt() != emulIt->hwPt()) {
        etGood = false;
         eventGood = false;
      }

      // object position mismatch (phi)
      if (dataIt->hwPhi() != emulIt->hwPhi()){
        posGood = false;
        eventGood = false;

      }

      // object position mismatch (eta)
      if (dataIt->hwEta() != emulIt->hwEta()) {
        posGood = false;
        eventGood = false;
      }

      // if both position and energy agree, jet is good
      if (etGood && posGood) {
        histograms.agreementSummary.fill(JETGOOD_S);
        histograms.jetSummary.fill(JETGOOD);
      } else {
        histograms.jetEtData.fill(dataIt->hwPt());
        histograms.jetEtaData.fill(dataIt->hwEta());
        histograms.jetPhiData.fill(dataIt->hwPhi());

        histograms.jetEtEmul.fill(emulIt->hwPt());
        histograms.jetEtaEmul.fill(emulIt->hwEta());
        histograms.jetPhiEmul.fill(emulIt->hwPhi());

        if (verbose) {
          edm::LogInfo("L1TdeStage2CaloLayer2") << "--- jet ---"<< std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data jet Et = "
                                                << dataIt->hwPt() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul jet Et = "
                                                << emulIt->hwPt() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data jet phi = "
                                                << dataIt->hwPhi() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul jet phi = "
                                                << emulIt->hwPhi() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data jet eta = "
                                                << dataIt->hwEta() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul jet eta = "
                                                << emulIt->hwEta() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "---"<< std::endl;
        }
      }

      // if only position agrees
      if (posGood && !etGood) {
        histograms.jetSummary.fill(JETETOFF);
      }

      // if only energy agrees
      if (!posGood && etGood) {
        histograms.jetSummary.fill(JETPOSOFF);
      }

      // keep track of jets
      histograms.agreementSummary.fill(NJETS_S);
      histograms.jetSummary.fill(NJETS);

      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
          emulIt == emulCol->end(currBx))
        break;
    }
  } else {
    if (dataCol->size(currBx) != 0 || emulCol->size(currBx) != 0)
      return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for e/gammas
bool L1TdeStage2CaloLayer2::compareEGs(
  const edm::Handle<l1t::EGammaBxCollection> & dataCol,
  const edm::Handle<l1t::EGammaBxCollection> & emulCol,
  const calolayer2dedqm::Histograms & histograms) const
{
  bool eventGood = true;

  l1t::EGammaBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::EGammaBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  // check length of collections
  if (dataCol->size(currBx) != emulCol->size(currBx)) {

    if (dataCol->size(currBx) < emulCol->size(currBx)) {

      // if only the data collection is empty, declare event as bad
      if (dataCol->isEmpty(currBx)) return false;

      // if there are more events in data loop over the data collection
      while (true) {
        if (dataIt->hwIso()) {
          histograms.isoEgEtData.fill(dataIt->hwPt());
          histograms.isoEgEtaData.fill(dataIt->hwEta());
          histograms.isoEgPhiData.fill(dataIt->hwPhi());
        } else {
          histograms.egEtData.fill(dataIt->hwPt());
          histograms.egEtaData.fill(dataIt->hwEta());
          histograms.egPhiData.fill(dataIt->hwPhi());
        }

        ++dataIt;

        if (dataIt == dataCol->end(currBx))
          break;
      }
    } else {

      // if only the emul collection is empty, declare event as bad
      if (emulCol->isEmpty(currBx)) return false;

      while (true) {
        if(emulIt->hwIso()) {
          histograms.isoEgEtEmul.fill(emulIt->hwPt());
          histograms.isoEgEtaEmul.fill(emulIt->hwEta());
          histograms.isoEgPhiEmul.fill(emulIt->hwPhi());
        } else {
          histograms.egEtEmul.fill(emulIt->hwPt());
          histograms.egEtaEmul.fill(emulIt->hwEta());
          histograms.egPhiEmul.fill(emulIt->hwPhi());
        }

        ++emulIt;

        if (emulIt == emulCol->end(currBx))
          break;
      }
    }

    histograms.problemSummary.fill(EGCOLLSIZE);
    return false;
  }

  // processing continues only of length of data collections is the same
  if (dataIt != dataCol->end(currBx) ||
      emulIt != emulCol->end(currBx)) {

    while(true) {

      bool posGood = true;
      bool etGood = true;
      bool iso = dataIt->hwIso();

      // object pt mismatch
      if (dataIt->hwPt() != emulIt->hwPt()) {
        etGood = false;
        eventGood = false;
      }

      // object position mismatch (phi)
      if (dataIt->hwPhi() != emulIt->hwPhi()) {
        posGood = false;
        eventGood = false;
      }

      // object position mismatch (eta)
      if (dataIt->hwEta() != emulIt->hwEta()) {
        posGood = false;
        eventGood = false;
      }

      // if both position and energy agree, object is good
      if (posGood && etGood) {
        histograms.agreementSummary.fill(EGGOOD_S);

        if (iso) {
          histograms.egSummary.fill(ISOEGGOOD);
        } else {
          histograms.egSummary.fill(EGGOOD);
        }

      } else {

        if (iso) {
          histograms.isoEgEtData.fill(dataIt->hwPt());
          histograms.isoEgEtaData.fill(dataIt->hwEta());
          histograms.isoEgPhiData.fill(dataIt->hwPhi());

          histograms.isoEgEtEmul.fill(emulIt->hwPt());
          histograms.isoEgEtaEmul.fill(emulIt->hwEta());
          histograms.isoEgPhiEmul.fill(emulIt->hwPhi());
        } else {
          histograms.egEtData.fill(dataIt->hwPt());
          histograms.egEtaData.fill(dataIt->hwEta());
          histograms.egPhiData.fill(dataIt->hwPhi());

          histograms.egEtEmul.fill(emulIt->hwPt());
          histograms.egEtaEmul.fill(emulIt->hwEta());
          histograms.egPhiEmul.fill(emulIt->hwPhi());
        }

        if (verbose) {
          edm::LogInfo("L1TdeStage2CaloLayer2") << "--- eg ---"<< std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data eg Et = "
                                                << dataIt->hwPt() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul eg Et = "
                                                << emulIt->hwPt() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data eg phi = "
                                                << dataIt->hwPhi() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul eg phi = "
                                                << emulIt->hwPhi() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data eg eta = "
                                                << dataIt->hwEta() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul eg eta = "
                                                << emulIt->hwEta() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "---"<< std::endl;
        }
      }

      // if only position agrees
      if (posGood && !etGood) {
        if (iso) {
          histograms.egSummary.fill(ISOEGETOFF);
        } else {
          histograms.egSummary.fill(EGETOFF);
        }
      }

      // if only energy agrees
      if (!posGood && etGood) {
        if (iso) {
          histograms.egSummary.fill(ISOEGPOSOFF);
        } else {
          histograms.egSummary.fill(EGPOSOFF);
        }
      }

      // keep track of number of objects
      if (iso) {
        histograms.egSummary.fill(NISOEGS);
      } else {
        histograms.egSummary.fill(NEGS);
      }
      histograms.agreementSummary.fill(NEGS_S);

      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
          emulIt == emulCol->end(currBx))
        break;
    }
  } else {
    if (dataCol->size(currBx) != 0 || emulCol->size(currBx) != 0)
      return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for taus
bool L1TdeStage2CaloLayer2::compareTaus(
  const edm::Handle<l1t::TauBxCollection> & dataCol,
  const edm::Handle<l1t::TauBxCollection> & emulCol,
  const calolayer2dedqm::Histograms & histograms) const
{
  bool eventGood = true;

  l1t::TauBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::TauBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  // check length of collections
  if (dataCol->size(currBx) != emulCol->size(currBx)) {

    if (dataCol->size(currBx) < emulCol->size(currBx)) {

      // if only the data collection is empty, declare event as bad
      if (dataCol->isEmpty(currBx)) return false;

      // if there are more events in data loop over the data collection
      while (true) {

        // Populate different set of histograms if object is solated

        if (dataIt->hwIso()) {
          histograms.isoTauEtData.fill(dataIt->hwPt());
          histograms.isoTauEtaData.fill(dataIt->hwEta());
          histograms.isoTauPhiData.fill(dataIt->hwPhi());
        } else {
          histograms.tauEtData.fill(dataIt->hwPt());
          histograms.tauEtaData.fill(dataIt->hwEta());
          histograms.tauPhiData.fill(dataIt->hwPhi());
        }

        ++dataIt;

        if (dataIt == dataCol->end(currBx))
          break;
      }
    } else {

      // if only the emul collection is bad, declare the event as bad
      if (emulCol->isEmpty(currBx)) return false;

      while (true) {

        // Populate different set of histograms if object is solated

        if(emulIt->hwIso()) {
          histograms.isoTauEtEmul.fill(emulIt->hwPt());
          histograms.isoTauEtaEmul.fill(emulIt->hwEta());
          histograms.isoTauPhiEmul.fill(emulIt->hwPhi());
        } else {
          histograms.tauEtEmul.fill(emulIt->hwPt());
          histograms.tauEtaEmul.fill(emulIt->hwEta());
          histograms.tauPhiEmul.fill(emulIt->hwPhi());
        }

        ++emulIt;

        if (emulIt == emulCol->end(currBx))
          break;
      }
    }

    histograms.problemSummary.fill(TAUCOLLSIZE);
    return false;
  }

  // processing continues only of length of data collections is the same
  if (dataIt != dataCol->end(currBx) ||
      emulIt != emulCol->end(currBx)) {

    while(true) {

      bool posGood = true;
      bool etGood = true;
      bool iso = dataIt->hwIso();

      // object Et mismatch
      if (dataIt->hwPt() != emulIt->hwPt()) {
        etGood = false;
        eventGood = false;
      }

      // object position mismatch (phi)
      if (dataIt->hwPhi() != emulIt->hwPhi()) {
        posGood = false;
        eventGood = false;
      }

      // object position mismatch (eta)
      if (dataIt->hwEta() != emulIt->hwEta()) {
        posGood = false;
        eventGood = false;
      }

      // if both position and energy agree, object is good
      if (posGood && etGood) {
        histograms.agreementSummary.fill(TAUGOOD_S);

        if (iso) {
          histograms.tauSummary.fill(ISOTAUGOOD);
        } else {
          histograms.tauSummary.fill(TAUGOOD);
        }
      } else {

        if (iso) {
          histograms.isoTauEtData.fill(dataIt->hwPt());
          histograms.isoTauEtaData.fill(dataIt->hwEta());
          histograms.isoTauPhiData.fill(dataIt->hwPhi());

          histograms.isoTauEtEmul.fill(emulIt->hwPt());
          histograms.isoTauEtaEmul.fill(emulIt->hwEta());
          histograms.isoTauPhiEmul.fill(emulIt->hwPhi());

        } else {
          histograms.tauEtData.fill(dataIt->hwPt());
          histograms.tauEtaData.fill(dataIt->hwEta());
          histograms.tauPhiData.fill(dataIt->hwPhi());

          histograms.tauEtEmul.fill(emulIt->hwPt());
          histograms.tauEtaEmul.fill(emulIt->hwEta());
          histograms.tauPhiEmul.fill(emulIt->hwPhi());
        }

        if (verbose) {
          edm::LogInfo("L1TdeStage2CaloLayer2") << "--- tau ---"<< std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data tau Et = "
                                                << dataIt->hwPt() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul tau Et = "
                                                << emulIt->hwPt() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data tau phi = "
                                                << dataIt->hwPhi() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul tau phi = "
                                                << emulIt->hwPhi() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "data tau eta = "
                                                << dataIt->hwEta() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "emul tau eta = "
                                                << emulIt->hwEta() << std::endl;
          edm::LogInfo("L1TdeStage2CaloLayer2") << "---"<< std::endl;
        }
      }

      // if only position agrees
      if (posGood && !etGood) {
        if (iso) {
          histograms.tauSummary.fill(ISOTAUETOFF);
        } else {
          histograms.tauSummary.fill(TAUETOFF);
        }
      }

      // if only energy agrees
      if (!posGood && etGood) {
        if (iso) {
          histograms.tauSummary.fill(ISOTAUPOSOFF);
        } else {
          histograms.tauSummary.fill(TAUPOSOFF);
        }
      }

      // keep track of number of objects
      if (iso) {
        histograms.tauSummary.fill(NISOTAUS);
      } else {
        histograms.tauSummary.fill(NTAUS);
      }

      histograms.agreementSummary.fill(NTAUS_S);

      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
          emulIt == emulCol->end(currBx))
        break;
    }
  } else {
    if (dataCol->size(currBx) != 0 || emulCol->size(currBx) != 0)
      return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for sums
bool L1TdeStage2CaloLayer2::compareSums(
  const edm::Handle<l1t::EtSumBxCollection> & dataCol,
  const edm::Handle<l1t::EtSumBxCollection> & emulCol,
  const calolayer2dedqm::Histograms & histograms) const
{
  bool eventGood = true;

  bool etGood = true;
  bool phiGood = true;

  double dataEt = 0;
  double emulEt = 0;
  double dataPhi = 0;
  double emulPhi = 0;

  l1t::EtSumBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::EtSumBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  // if either data or emulator collections are empty, or they have different
  // size, mark the event as bad (this should never occur in normal running)
  if (dataCol->isEmpty(currBx) || emulCol->isEmpty(currBx) ||
      (dataCol->size(currBx) != emulCol->size(currBx)))
    return false;

  while(true) {

    // It should be possible to implement this with a switch statement
    etGood = true;
    phiGood = true;

    // ETT
    if (l1t::EtSum::EtSumType::kTotalEt == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      if (dataEt != emulEt) {
        eventGood = false;

        histograms.ettData.fill(dataEt);
        histograms.ettEmul.fill(emulEt);
      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(ETTSUMGOOD);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "ETT       | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NETTSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // ETTEM
    if (l1t::EtSum::EtSumType::kTotalEtEm == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      if (dataEt != emulEt) {
        eventGood = false;
        histograms.ettEmData.fill(dataEt);
        histograms.ettEmEmul.fill(emulEt);

      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(ETTSUMGOOD);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "ETTEM     | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NETTSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // HTT
    if (l1t::EtSum::EtSumType::kTotalHt == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      if (dataEt != emulEt) {
        eventGood = false;
        histograms.httData.fill(dataEt);
        histograms.httEmul.fill(emulEt);
      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(HTTSUMGOOD);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "HTT       | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NHTTSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // MET
    if (l1t::EtSum::EtSumType::kMissingEt == dataIt->getType()
        && dataIt->hwPt() != 0) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      dataPhi = dataIt->hwPhi();
      emulPhi = emulIt->hwPhi();

      if (dataEt != emulEt) {
        etGood = false;
        eventGood = false;
      }

      if (dataPhi != emulPhi) {
        phiGood = false;
        eventGood = false;
      }

      if (etGood && phiGood) {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(METSUMGOOD);
      } else {
        histograms.metEtData.fill(dataEt);
        histograms.metPhiData.fill(dataPhi);
        histograms.metEtEmul.fill(emulEt);
        histograms.metPhiEmul.fill(emulPhi);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "MET       | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;

        edm::LogInfo("L1TdeStage2CaloLayer2") << "MET phi   | ";
        if (dataPhi != emulPhi)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataPhi << "\t" << emulPhi;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NMETSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // METHF
    if (l1t::EtSum::EtSumType::kMissingEtHF == dataIt->getType()
        && dataIt->hwPt() != 0) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      dataPhi = dataIt->hwPhi();
      emulPhi = emulIt->hwPhi();

      if (dataEt != emulEt) {
        etGood = false;
        eventGood = false;
      }

      if (dataPhi != emulPhi) {
        phiGood = false;
        eventGood = false;
      }

      if (etGood && phiGood) {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(METSUMGOOD);
      } else {
        histograms.metHFEtData.fill(dataEt);
        histograms.metHFPhiData.fill(dataPhi);
        histograms.metHFEtEmul.fill(emulEt);
        histograms.metHFPhiEmul.fill(emulPhi);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "METHF     | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;

        edm::LogInfo("L1TdeStage2CaloLayer2") << "METHF phi | ";
        if (dataPhi != emulPhi)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataPhi << "\t" << emulPhi;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NMETSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // MHT
    if (l1t::EtSum::EtSumType::kMissingHt == dataIt->getType()
        && dataIt->hwPt() != 0) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      dataPhi = dataIt->hwPhi();
      emulPhi = emulIt->hwPhi();

      if (dataEt != emulEt) {
        etGood = false;
        eventGood = false;
      }

      if (!(etGood && dataEt == 0)) {
        if (dataPhi != emulPhi) {
          phiGood = false;
          eventGood = false;
        }
      }

      if (etGood && phiGood) {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(MHTSUMGOOD);
      } else {
        histograms.mhtEtData.fill(dataEt);
        histograms.mhtPhiData.fill(dataPhi);
        histograms.mhtEtEmul.fill(emulEt);
        histograms.mhtPhiEmul.fill(emulPhi);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "MHT       | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;

        edm::LogInfo("L1TdeStage2CaloLayer2") << "MHT phi   | ";
        if (dataPhi != emulPhi)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataPhi << "\t" << emulPhi;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NMHTSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // MHTHF
    if (l1t::EtSum::EtSumType::kMissingHtHF == dataIt->getType()
        && dataIt->hwPt() != 0) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      dataPhi = dataIt->hwPhi();
      emulPhi = emulIt->hwPhi();

      if (dataEt != emulEt) {
        phiGood = false;
        eventGood = false;
      }

      if (!(etGood && dataEt == 0)) {
        if (dataPhi != emulPhi) {
          phiGood = false;
          eventGood = false;
        }
      }

      if (etGood && phiGood) {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(MHTSUMGOOD);
      } else {
        histograms.mhtHFEtData.fill(dataEt);
        histograms.mhtHFPhiData.fill(dataPhi);
        histograms.mhtHFEtEmul.fill(emulEt);
        histograms.mhtHFPhiEmul.fill(emulPhi);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "MHTHF     | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;

        edm::LogInfo("L1TdeStage2CaloLayer2") << "MHTHF phi | ";
        if (dataPhi != emulPhi)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataPhi << "\t" << emulPhi;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NMHTSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // MBHFP0
    if (l1t::EtSum::EtSumType::kMinBiasHFP0 == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      if (dataEt != emulEt) {
        eventGood = false;
        histograms.mbhfp0Data.fill(dataEt);
        histograms.mbhfp0Emul.fill(emulEt);
      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(MBHFSUMGOOD);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "MBHFP0    | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NMBHFSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // MBHFM0
    if (l1t::EtSum::EtSumType::kMinBiasHFM0 == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      if (dataEt != emulEt) {
        eventGood = false;
        histograms.mbhfm0Data.fill(dataEt);
        histograms.mbhfm0Emul.fill(emulEt);
      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(MBHFSUMGOOD);
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NMBHFSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // MBHFP1
    if (l1t::EtSum::EtSumType::kMinBiasHFP1 == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      if (dataEt != emulEt) {
        eventGood = false;
        histograms.mbhfp1Data.fill(dataEt);
        histograms.mbhfp1Emul.fill(emulEt);
      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(MBHFSUMGOOD);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "MBHFP1    | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NMBHFSUMS);
      histograms.sumSummary.fill(NSUMS);
    }

    // MBHFM1
    if (l1t::EtSum::EtSumType::kMinBiasHFM1 == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      histograms.sumSummary.fill(NMBHFSUMS);

      if (dataEt != emulEt) {
        eventGood = false;
        histograms.mbhfm1Data.fill(dataEt);
        histograms.mbhfm1Emul.fill(emulEt);
      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(MBHFSUMGOOD);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "MBHFM1    | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NSUMS);
    }

    // TowerCount
    if (l1t::EtSum::EtSumType::kTowerCount == dataIt->getType()) {

      dataEt = dataIt->hwPt();
      emulEt = emulIt->hwPt();

      if (dataEt != emulEt) {
        eventGood = false;
        histograms.towCountData.fill(dataEt);
        histograms.towCountEmul.fill(emulEt);
      } else {
        histograms.agreementSummary.fill(SUMGOOD_S);
        histograms.sumSummary.fill(SUMGOOD);
        histograms.sumSummary.fill(TOWCOUNTGOOD);
      }

      if (verbose) {
        edm::LogInfo("L1TdeStage2CaloLayer2") << "TowCount  | ";
        if (dataEt != emulEt)
          edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
        else
          edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";
        edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
        edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
      }

      // update sum counters
      histograms.agreementSummary.fill(NSUMS_S);
      histograms.sumSummary.fill(NTOWCOUNTS);
      histograms.sumSummary.fill(NSUMS);
    }

    ++dataIt;
    ++emulIt;

    if (dataIt == dataCol->end(currBx) || emulIt == emulCol->end(currBx))
      break;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

