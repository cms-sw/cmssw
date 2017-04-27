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

void L1TdeStage2CaloLayer2::endLuminosityBlock (const edm::LuminosityBlock&,
					       const edm::EventSetup&)
{
  double totRatio = totalEvents != 0 ? goodEvents / (totalEvents * 1.0) : 0;
  double jetRatio = totalJets != 0 ? goodJets / (totalJets * 1.0) : 0;
  double egRatio = (totalEGs + totalIsoEGs) != 0 ?
    (goodEGs + goodIsoEGs) / ((totalEGs + totalIsoEGs) * 1.0) : 0;
  double tauRatio = (totalTaus + totalIsoTaus) != 0 ?
    (goodTaus + goodIsoTaus) / ((totalTaus + totalIsoTaus) * 1.0) : 0;
  double sumRatio = totalSums != 0 ? goodSums / (totalSums * 1.0) : 0;

  // divide the bin contents of high-level summary histogram by the totals
  // this might require too much memory (called at end of run)
  TH1F * hlSum = agreementSummary->getTH1F();
  hlSum->SetBinContent(hlSum->FindBin(EVENTGOOD), totRatio);
  hlSum->SetBinContent(hlSum->FindBin(JETGOOD_S), jetRatio);
  hlSum->SetBinContent(hlSum->FindBin(EGGOOD_S), egRatio);
  hlSum->SetBinContent(hlSum->FindBin(TAUGOOD_S), tauRatio);
  hlSum->SetBinContent(hlSum->FindBin(SUMGOOD_S), sumRatio);

  // this might require too much memory (called at end of run)
  double posOffJetRatio = totalJets != 0 ? posOffJets / (totalJets * 1.0) : 0;
  double etOffJetRatio = totalJets != 0 ? etOffJets / (totalJets * 1.0) : 0;

  TH1F * jetHist = jetSummary->getTH1F();
  jetHist->SetBinContent(jetHist->FindBin(JETGOOD), jetRatio);
  jetHist->SetBinContent(jetHist->FindBin(JETPOSOFF), posOffJetRatio);
  jetHist->SetBinContent(jetHist->FindBin(JETETOFF), etOffJetRatio);

  double goodNonIsoEGRatio = totalEGs != 0 ? goodEGs / (totalEGs * 1.0) : 0;
  double posOffNonIsoEGRatio = totalEGs != 0 ? posOffEGs / (totalEGs * 1.0) : 0;
  double etOffNonIsoEGRatio = totalEGs != 0 ? etOffEGs / (totalEGs * 1.0) : 0;

  double goodIsoEGRatio = totalIsoEGs != 0 ?
    goodIsoEGs / (totalIsoEGs * 1.0) : 0;
  double posOffIsoEGRatio = totalIsoEGs != 0 ?
    posOffIsoEGs / (totalIsoEGs * 1.0) : 0;
  double etOffIsoEGRatio = totalIsoEGs != 0 ?
    etOffIsoEGs / (totalIsoEGs * 1.0) : 0;

  TH1F * egHist = egSummary->getTH1F();
  egHist->SetBinContent(egHist->FindBin(EGGOOD), goodNonIsoEGRatio);
  egHist->SetBinContent(egHist->FindBin(EGPOSOFF), posOffNonIsoEGRatio);
  egHist->SetBinContent(egHist->FindBin(EGETOFF), etOffNonIsoEGRatio);
  egHist->SetBinContent(egHist->FindBin(ISOEGGOOD), goodIsoEGRatio);
  egHist->SetBinContent(egHist->FindBin(ISOEGPOSOFF), posOffIsoEGRatio);
  egHist->SetBinContent(egHist->FindBin(ISOEGETOFF), etOffIsoEGRatio);

  double goodNonIsoTauRatio = totalTaus != 0 ? goodTaus / (totalTaus * 1.0) : 0;
  double posOffNonIsoTauRatio = totalTaus != 0 ?
    posOffTaus / (totalTaus * 1.0) : 0;
  double etOffNonIsoTauRatio = totalTaus != 0 ?
    etOffTaus / (totalTaus * 1.0) : 0;
  double goodIsoTauRatio = totalIsoTaus != 0 ?
    goodIsoTaus / (totalIsoTaus * 1.0) : 0;
  double posOffIsoTauRatio = totalIsoTaus != 0 ?
    posOffIsoTaus / (totalIsoTaus * 1.0) : 0;
  double etOffIsoTauRatio = totalIsoTaus != 0 ?
    etOffIsoTaus / (totalIsoTaus * 1.0) : 0;

  TH1F * tauHist = tauSummary->getTH1F();
  tauHist->SetBinContent(tauHist->FindBin(TAUGOOD), goodNonIsoTauRatio);
  tauHist->SetBinContent(tauHist->FindBin(TAUPOSOFF), posOffNonIsoTauRatio);
  tauHist->SetBinContent(tauHist->FindBin(TAUETOFF), etOffNonIsoTauRatio);
  tauHist->SetBinContent(tauHist->FindBin(ISOTAUGOOD), goodIsoTauRatio);
  tauHist->SetBinContent(tauHist->FindBin(ISOTAUPOSOFF), posOffIsoTauRatio);
  tauHist->SetBinContent(tauHist->FindBin(ISOTAUETOFF), etOffIsoTauRatio);

  double ettRatio = totalETTSums != 0 ? goodETTSums / (totalETTSums * 1.0) : 0;
  double httRatio = totalHTTSums != 0 ? goodHTTSums / (totalHTTSums * 1.0) : 0;
  double metRatio = totalMETSums != 0 ? goodMETSums / (totalMETSums * 1.0) : 0;
  double mhtRatio = totalMHTSums != 0 ? goodMHTSums / (totalMHTSums * 1.0) : 0;
  double mbhfRatio = totalMBHFSums != 0 ?
    goodMBHFSums / (totalMBHFSums * 1.0) : 0;
  double towRatio = totalTowCountSums != 0 ?
    goodTowCountSums / (totalTowCountSums * 1.0) : 0;

  TH1F * sumHist = sumSummary->getTH1F();
  sumHist->SetBinContent(sumHist->FindBin(SUMGOOD), sumRatio);
  sumHist->SetBinContent(sumHist->FindBin(ETTSUMGOOD), ettRatio);
  sumHist->SetBinContent(sumHist->FindBin(HTTSUMGOOD), httRatio);
  sumHist->SetBinContent(sumHist->FindBin(METSUMGOOD), metRatio);
  sumHist->SetBinContent(sumHist->FindBin(MHTSUMGOOD), mhtRatio);
  sumHist->SetBinContent(sumHist->FindBin(MBHFSUMGOOD), mbhfRatio);
  sumHist->SetBinContent(sumHist->FindBin(TOWCOUNTGOOD), towRatio);
}

void L1TdeStage2CaloLayer2::bookHistograms(
  DQMStore::IBooker &ibooker,
  edm::Run const &,
  edm::EventSetup const&) {

  // DQM directory to store histograms with problematic jets
  ibooker.setCurrentFolder(monitorDir + "/Problematic Jets candidates");

  jetEtData = ibooker.book1D("Problematic Data Jet iEt", "Jet iE_{T}",
			     1400, 0, 1399);
  jetEtaData = ibooker.book1D("Problematic Data Jet iEta", "Jet i#eta",
			  227, -113.5, 113.5);
  jetPhiData = ibooker.book1D("Problematic Data Jet iPhi", "Jet i#phi",
			  288, -0.5, 143.5);
  jetEtEmul = ibooker.book1D("Problematic Emul Jet iEt", "Jet iE_{T}",
			     1400, 0, 1399);
  jetEtaEmul = ibooker.book1D("Problematic Emul Jet iEta", "Jet i#eta",
			  227, -113.5, 113.5);
  jetPhiEmul = ibooker.book1D("Problematic Emul Jet iPhi", "Jet i#phi",
			  288, -0.5, 143.5);

  // DQM directory to store histograms with problematic e/gs
  ibooker.setCurrentFolder(monitorDir + "/Problematic EG candidtes");

  egEtData = ibooker.book1D("Problematic Data Eg iEt", "Eg iE_{T}",
			    1400, 0, 1399);
  egEtaData = ibooker.book1D("Problematic Data Eg iEta", "Eg i#eta",
			     227, -113.5, 113.5);
  egPhiData = ibooker.book1D("Problematic Data Eg iPhi", "Eg i#phi",
			     288, -0.5, 143.5);
  egEtEmul = ibooker.book1D("Problematic Emul Eg iEt", "Eg iE_{T}",
			    1400, 0, 1399);
  egEtaEmul = ibooker.book1D("Problematic Emul Eg iEta", "Eg i#eta",
			     227, -113.5, 113.5);
  egPhiEmul = ibooker.book1D("Problematic Emul Eg iPhi", "Eg i#phi",
			     288, -0.5, 143.5);

  isoEgEtData = ibooker.book1D("Problematic Isolated Data Eg iEt",
			       "Iso Eg iE_{T}", 1400, 0, 1399);
  isoEgEtaData = ibooker.book1D("Problematic Isolated Data Eg iEta",
				"Iso Eg i#eta", 227, -113.5, 113.5);
  isoEgPhiData = ibooker.book1D("Problematic Isolated Data Eg iPhi",
				"Iso Eg i#phi", 288, -0.5, 143.5);
  isoEgEtEmul = ibooker.book1D("Problematic Isolated Emul Eg iEt",
			       "Iso Eg iE_{T}", 1400, 0, 1399);
  isoEgEtaEmul = ibooker.book1D("Problematic Isolated Emul Eg iEta",
				"Iso Eg i#eta", 227, -113.5, 113.5);
  isoEgPhiEmul = ibooker.book1D("Problematic Isolated Emul Eg iPhi",
				"Iso Eg i#phi", 288, -0.5, 143.5);

  // DQM directory to store histograms with problematic taus
  ibooker.setCurrentFolder(monitorDir + "/Problematic Tau candidtes");

  tauEtData = ibooker.book1D("Problematic Data Tau iEt", "Tau iE_{T}",
			     1400, 0, 1399);
  tauEtaData = ibooker.book1D("Problematic Data Tau iEta", "Tau i#eta",
			  227, -113.5, 113.5);
  tauPhiData = ibooker.book1D("Problematic Data Tau iPhi", "Tau i#phi",
			  288, -0.5, 143.5);
  tauEtEmul = ibooker.book1D("Problematic Emul Tau iEt", "Tau iE_{T}",
			     1400, 0, 1399);
  tauEtaEmul = ibooker.book1D("Problematic Emul Tau iEta", "Tau i#eta",
			  227, -113.5, 113.5);
  tauPhiEmul = ibooker.book1D("Problematic Emul Tau iPhi", "Tau i#phi",
			  288, -0.5, 143.5);

  isoTauEtData = ibooker.book1D("Problematic Isolated Data Tau iEt",
				"Iso Tau iE_{T}", 1400, 0, 1399);
  isoTauEtaData = ibooker.book1D("Problematic Isolated Data Tau iEta",
				 "Iso Tau i#eta", 227, -113.5, 113.5);
  isoTauPhiData = ibooker.book1D("Problematic Isolated Data Tau iPhi",
				 "Iso Tau i#phi", 288, -0.5, 143.5);
  isoTauEtEmul = ibooker.book1D("Problematic Isolated Emul Tau iEt",
				"Iso Tau iE_{T}", 1400, 0, 1399);
  isoTauEtaEmul = ibooker.book1D("Problematic Isolated Emul Tau iEta",
				 "Iso Tau i#eta", 227, -113.5, 113.5);
  isoTauPhiEmul = ibooker.book1D("Problematic Isolated Emul Tau iPhi",
				 "Iso Tau i#phi", 288, -0.5, 143.5);

  // DQM directory to store histograms with problematic sums
  ibooker.setCurrentFolder(monitorDir + "/Problematic Sums");

  // book ETT type sums
  ettData = ibooker.book1D("Problematic ETT Sum - Data", "ETT iE_{T}",
			   7000, -0.5, 6999.5);
  ettEmul = ibooker.book1D("Problematic ETT Sum - Emulator", "ETT iE_{T}",
			   7000, -0.5, 6999.5);
  ettHFData = ibooker.book1D("Problematic ETTHF Sum - Data", "ETTHF iE_{T}",
			   7000, -0.5, 6999.5);
  ettHFEmul = ibooker.book1D("Problematic ETTHF Sum - Emulator", "ETTHF iE_{T}",
			   7000, -0.5, 6999.5);
  ettEmData = ibooker.book1D("Problematic ETTEM Sum - Data", "ETTEM iE_{T}",
			   7000, -0.5, 6999.5);
  ettEmEmul = ibooker.book1D("Problematic ETTEM Sum - Emulator", "ETTEM iE_{T}",
			   7000, -0.5, 6999.5);

  // book HTT type sums
  httData = ibooker.book1D("Problematic HTT Sum - Data", "HTT iE_{T}",
			   7000, -0.5, 6999.5);
  httEmul = ibooker.book1D("Problematic HTT Sum - Emulator", "HTT iE_{T}",
			   7000, -0.5, 6999.5);
  httHFData = ibooker.book1D("Problematic HTTHF Sum - Data", "HTTHF iE_{T}",
			   7000, -0.5, 6999.5);
  httHFEmul = ibooker.book1D("Problematic HTTHF Sum - Emulator", "HTTHF iE_{T}",
			   7000, -0.5, 6999.5);

  // book MET type sums
  metEtData = ibooker.book1D("Problematic MET Sum Et - Data", "MET iE_{T}",
			     7000, -0.5, 6999.5);
  metEtEmul = ibooker.book1D("Problematic MET Sum Et - Emulator", "MET iE_{T}",
			     7000, -0.5, 6999.5);
  metPhiData = ibooker.book1D("Problematic MET Sum phi - Data", "MET i#phi",
			      1008, -0.5, 1007.5);
  metPhiEmul = ibooker.book1D("Problematic MET Sum phi - Emulator", "MET i#phi",
			      1008, -0.5, 1007.5);

  metHFEtData = ibooker.book1D("Problematic METHF Sum Et - Data",
			       "METHF iE_{T}", 7000, -0.5, 6999.5);
  metHFEtEmul = ibooker.book1D("Problematic METHF Sum Et - Emulator",
			       "METHF iE_{T}", 7000, -0.5, 6999.5);
  metHFPhiData = ibooker.book1D("Problematic METHF Sum phi - Data",
				"METHF i#phi", 1008, -0.5, 1007.5);
  metHFPhiEmul = ibooker.book1D("Problematic METHF Sum phi - Emulator",
				"METHF i#phi", 1008, -0.5, 1007.5);

  // book MHT type sums
  mhtEtData = ibooker.book1D("Problematic MHT Sum Et - Data", "MHT iE_{T}",
			     7000, -0.5, 6999.5);
  mhtEtEmul = ibooker.book1D("Problematic MHT Sum Et - Emulator", "MHT iE_{T}",
			     7000, -0.5, 6999.5);
  mhtPhiData = ibooker.book1D("Problematic MHT Sum phi - Data", "MHT i#phi",
			      1008, -0.5, 1007.5);
  mhtPhiEmul = ibooker.book1D("Problematic MHT Sum phi - Emulator", "MHT i#phi",
			      1008, -0.5, 1007.5);

  mhtHFEtData = ibooker.book1D("Problematic MHTHF Sum Et - Data",
			       "MHTHF iE_{T}", 7000, -0.5, 6999.5);
  mhtHFEtEmul = ibooker.book1D("Problematic MHTHF Sum Et - Emulator",
			       "MHTHF iE_{T}", 7000, -0.5, 6999.5);
  mhtHFPhiData = ibooker.book1D("Problematic MHTHF Sum phi - Data",
				"MHTHF i#phi", 1008, -0.5, 1007.5);
  mhtHFPhiEmul = ibooker.book1D("Problematic MHTHF Sum phi - Emulator",
				"MHTHF i#phi", 1008, -0.5, 1007.5);

  // book minimum bias sums
  mbhfp0Data = ibooker.book1D("Problematic MBHFP0 Sum - Data",
			      "", 16, -0.5, 15.5);
  mbhfp0Emul = ibooker.book1D("Problematic MBHFP0 Sum - Emulator",
			      "", 16, -0.5, 15.5);
  mbhfm0Data = ibooker.book1D("Problematic MBHFM0 Sum - Data",
			      "", 16, -0.5, 15.5);
  mbhfm0Emul = ibooker.book1D("Problematic MBHFM0 Sum - Emulator",
			      "", 16, -0.5, 15.5);
  mbhfm1Data = ibooker.book1D("Problematic MBHFM1 Sum - Data",
			      "", 16, -0.5, 15.5);
  mbhfm1Emul = ibooker.book1D("Problematic MBHFM1 Sum - Emulator",
			      "", 16, -0.5, 15.5);
  mbhfp1Data = ibooker.book1D("Problematic MBHFP1 Sum - Data",
			      "", 16, -0.5, 15.5);
  mbhfp1Emul = ibooker.book1D("Problematic MBHFP1 Sum - Emulator",
			      "", 16, -0.5, 15.5);

  // book tower count sums
  towCountData = ibooker.book1D("Problematic Tower Count Sum - Data",
				"", 5904, -0.5, 5903.5);
  towCountEmul = ibooker.book1D("Problematic Tower Count Sum - Emulator",
				"", 5904, -0.5, 5903.5);
  // for reference on arguments of book2D, see
  // https://cmssdt.cern.ch/SDT/doxygen/CMSSW_8_0_24/doc/html/df/d26/DQMStore_8cc_source.html#l01070


  // setup the directory where the histograms are to be visualised, value is set
  // in constructor and taken from python configuration file for module
  ibooker.setCurrentFolder(monitorDir);

  // Jet energy in MP firmware is stored in 16 bits which sets the range of
  // jet energy to 2^16 * 0.5 GeV = 32768 GeV (65536 hardware units)
  // --- this is only for MP jets, the demux jets have much decreased precision
  // --- and this should be replaced

  // the index of the first bin in histogram should match value of first enum
  agreementSummary = ibooker.book1D(
    "CaloL2 Object Agreement Summary",
    "CaloL2 event-by-event object agreement fractions", 5, 1, 6);

  agreementSummary->getTH1F()->GetXaxis()->SetBinLabel(EVENTGOOD,
						       "good events");
  agreementSummary->getTH1F()->GetXaxis()->SetBinLabel(JETGOOD_S, "good jets");
  agreementSummary->getTH1F()->GetXaxis()->SetBinLabel(EGGOOD_S, "good e/gs");
  agreementSummary->getTH1F()->GetXaxis()->SetBinLabel(TAUGOOD_S, "good taus");
  agreementSummary->getTH1F()->GetXaxis()->SetBinLabel(SUMGOOD_S, "good sums");
  jetSummary = ibooker.book1D(
    "Jet Agreement Summary", "Jet Agreement Summary", 3, 1, 4);
  jetSummary->getTH1F()->GetXaxis()->SetBinLabel(JETGOOD, "good jets");
  jetSummary->getTH1F()->GetXaxis()->SetBinLabel(JETPOSOFF,
						 "jets pos off only");
  jetSummary->getTH1F()->GetXaxis()->SetBinLabel(JETETOFF, "jets Et off only ");

  egSummary = ibooker.book1D(
    "EG Agreement Summary", "EG Agreement Summary", 6, 1, 7);
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(EGGOOD, "good e/gs");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(EGPOSOFF, "e/gs pos off");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(EGETOFF, "e/gs Et off");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(ISOEGGOOD, "good iso e/gs");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(ISOEGPOSOFF,
						"iso e/gs pos off");
  egSummary->getTH1F()->GetXaxis()->SetBinLabel(ISOEGETOFF, "iso e/gs Et off");

  tauSummary = ibooker.book1D(
    "Tau Agreement Summary", "Tau Agreement Summary", 6, 1, 7);
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(TAUGOOD, "good taus");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(TAUPOSOFF, "taus pos off");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(TAUETOFF, "taus Et off");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(ISOTAUGOOD, "good iso taus");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(ISOTAUPOSOFF,
						 "iso taus pos off");
  tauSummary->getTH1F()->GetXaxis()->SetBinLabel(ISOTAUETOFF,
						 "iso taus Et off");

  sumSummary = ibooker.book1D(
    "Emergy Sum Agreement Summary", "Sum Agreement Summary", 7, 1, 8);
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(SUMGOOD, "good sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(ETTSUMGOOD, "good ETT sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(HTTSUMGOOD, "good HTT sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(METSUMGOOD, "good MET sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(MHTSUMGOOD, "good MHT sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(MBHFSUMGOOD, "good MBHF sums");
  sumSummary->getTH1F()->GetXaxis()->SetBinLabel(TOWCOUNTGOOD,
						 "good TowCount sums");
}
void L1TdeStage2CaloLayer2::analyze (
  const edm::Event& e,
  const edm::EventSetup & c) {

  ++totalEvents;

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

  TH1F * agreementHist = agreementSummary->getTH1F();

  l1t::EGammaBxCollection::const_iterator dataEGIt = egDataCol->begin(currBx);
  l1t::EGammaBxCollection::const_iterator emulEGIt = egEmulCol->begin(currBx);

  l1t::TauBxCollection::const_iterator dataTauIt = tauDataCol->begin(currBx);
  l1t::TauBxCollection::const_iterator emulTauIt = tauEmulCol->begin(currBx);

  l1t::EtSumBxCollection::const_iterator dataSumIt = sumDataCol->begin(currBx);
  l1t::EtSumBxCollection::const_iterator emulSumIt = sumEmulCol->begin(currBx);

  if (!compareJets(jetDataCol, jetEmulCol)) {
    eventGood = false;
  }

  if (!compareEGs(egDataCol, egEmulCol)) {
    eventGood = false;
  }

  if (!compareTaus(tauDataCol, tauEmulCol)) {
    eventGood = false;
  }

  if (!compareSums(sumDataCol, sumEmulCol)) {
    eventGood = false;
  }

  /**
     Questions:
     - what could make the data and emul bx ranges to be different?
     - how can I confirm that the emulator data is being filled?
  */

  if (eventGood) {
    agreementHist->Fill(EVENTGOOD);
    ++goodEvents;
  }
}

// comparison method for jets
bool L1TdeStage2CaloLayer2::compareJets(
  const edm::Handle<l1t::JetBxCollection> & dataCol,
  const edm::Handle<l1t::JetBxCollection> & emulCol)
{
  bool eventGood = true;

  l1t::JetBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::JetBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  TH1F * objEtHistData = jetEtData->getTH1F();
  TH1F * objPhiHistData = jetPhiData->getTH1F();
  TH1F * objEtaHistData = jetEtaData->getTH1F();

  TH1F * objEtHistEmul = jetEtEmul->getTH1F();
  TH1F * objPhiHistEmul = jetPhiEmul->getTH1F();
  TH1F * objEtaHistEmul = jetEtaEmul->getTH1F();

  TH1F * objSummaryHist = jetSummary->getTH1F();
  TH1F * summaryHist = agreementSummary->getTH1F();

  // process jets
  if (dataCol->size() != emulCol->size()) {

    if (dataCol->size() < emulCol->size()) {
      if (dataCol->size() < 1) {
	return false;
      }

      while (true) {
	objEtHistData->Fill(dataIt->hwPt());
	objEtaHistData->Fill(dataIt->hwEta());
	objPhiHistData->Fill(dataIt->hwPhi());

	++dataIt;

	if (dataIt == dataCol->end(currBx))
	  break;
      }
    } else {

      if (emulCol->size() < 1) {
	return false;
      }

      while (true) {

	objEtHistEmul->Fill(emulIt->hwPt());
	objEtaHistEmul->Fill(emulIt->hwEta());
	objPhiHistEmul->Fill(emulIt->hwPhi());

	++emulIt;

	if (emulIt == emulCol->end(currBx))
	  break;
      }
    }

    return false;
  }

  if (dataIt != dataCol->end(currBx) ||
      emulIt != emulCol->end(currBx)) {
    while(true) {

      bool posGood = true;
      bool etGood = true;

      // jet Et mismatch
      if (dataIt->hwPt() != emulIt->hwPt()) {
        etGood = false;
 	eventGood = false;
      }

      // jet position mismatch
      if (dataIt->hwPhi() != emulIt->hwPhi()){
	posGood = false;
	eventGood = false;
      }
      if (dataIt->hwEta() != emulIt->hwEta()) {
	posGood = false;
	eventGood = false;
      }

      // if both position and energy agree, jet is good
      if (eventGood) {
	summaryHist->Fill(JETGOOD_S);
	objSummaryHist->Fill(JETGOOD);
	++goodJets;
      } else {
	objEtHistData->Fill(dataIt->hwPt());
	objEtaHistData->Fill(dataIt->hwEta());
	objPhiHistData->Fill(dataIt->hwPhi());

	objEtHistEmul->Fill(emulIt->hwPt());
	objEtaHistEmul->Fill(emulIt->hwEta());
	objPhiHistEmul->Fill(emulIt->hwPhi());
      }

      // if only position agrees
      if (posGood && !etGood) {
	objSummaryHist->Fill(JETETOFF);

	++etOffJets;
      }

      // if only energy agrees
      if (!posGood && etGood) {
	objSummaryHist->Fill(JETPOSOFF);
	++posOffJets;
      }

      // keep track of jets
      ++totalJets;

      // increase position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
	  emulIt == emulCol->end(currBx))
	break;
    }
  } else {

    return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for e/gammas
bool L1TdeStage2CaloLayer2::compareEGs(
  const edm::Handle<l1t::EGammaBxCollection> & dataCol,
  const edm::Handle<l1t::EGammaBxCollection> & emulCol)
{
  bool eventGood = true;

  l1t::EGammaBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::EGammaBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  TH1F * isoObjEtHistData = isoEgEtData->getTH1F();
  TH1F * isoObjPhiHistData = isoEgPhiData->getTH1F();
  TH1F * isoObjEtaHistData = isoEgEtaData->getTH1F();
  TH1F * isoObjEtHistEmul = isoEgEtEmul->getTH1F();
  TH1F * isoObjPhiHistEmul = isoEgPhiEmul->getTH1F();
  TH1F * isoObjEtaHistEmul = isoEgEtaEmul->getTH1F();

  TH1F * objEtHistData = egEtData->getTH1F();
  TH1F * objPhiHistData = egPhiData->getTH1F();
  TH1F * objEtaHistData = egEtaData->getTH1F();
  TH1F * objEtHistEmul = egEtEmul->getTH1F();
  TH1F * objPhiHistEmul = egPhiEmul->getTH1F();
  TH1F * objEtaHistEmul = egEtaEmul->getTH1F();

  TH1F * objSummaryHist = egSummary->getTH1F();
  TH1F * summaryHist = agreementSummary->getTH1F();

  // check length of collections
  if (dataCol->size() != emulCol->size()) {

    if (dataCol->size() < emulCol->size()) {
      if (dataCol->size() < 1)
	return false;

      // if there are more events in data loop over the data collection
      while (true) {

	// Populate different set of histograms if object is solated
	if (dataIt->hwIso()) {
	  isoObjEtHistData->Fill(dataIt->hwPt());
	  isoObjEtaHistData->Fill(dataIt->hwEta());
	  isoObjPhiHistData->Fill(dataIt->hwPhi());
	} else {
	  objEtHistData->Fill(dataIt->hwPt());
	  objEtaHistData->Fill(dataIt->hwEta());
	  objPhiHistData->Fill(dataIt->hwPhi());
	}

	++dataIt;

	if (dataIt == dataCol->end(currBx))
	  break;
      }
    } else {

      if (emulCol->size() < 1)
	return false;

      while (true) {

	// Populate different set of histograms if object is solated

	if(emulIt->hwIso()) {
	  isoObjEtHistEmul->Fill(emulIt->hwPt());
	  isoObjEtaHistEmul->Fill(emulIt->hwEta());
	  isoObjPhiHistEmul->Fill(emulIt->hwPhi());
	} else {
	  objEtHistEmul->Fill(emulIt->hwPt());
	  objEtaHistEmul->Fill(emulIt->hwEta());
	  objPhiHistEmul->Fill(emulIt->hwPhi());
	}

	++emulIt;

	if (emulIt == emulCol->end(currBx))
	  break;
      }
    }

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

      // object position mismatch
      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	posGood = false;
	eventGood = false;
      }
      if (dataIt->hwEta() != emulIt->hwEta()) {
	posGood = false;
	eventGood = false;
      }

      // if both position and energy agree, object is good
      if (eventGood) {
	summaryHist->Fill(EGGOOD_S);

	if (iso) {
	  ++goodIsoEGs;
	  objSummaryHist->Fill(ISOEGGOOD);
	} else {
	  ++goodEGs;
	  objSummaryHist->Fill(EGGOOD);
	}

      } else {

	if (iso) {
	  isoObjEtHistData->Fill(dataIt->hwPt());
	  isoObjEtaHistData->Fill(dataIt->hwEta());
	  isoObjPhiHistData->Fill(dataIt->hwPhi());

	  isoObjEtHistEmul->Fill(emulIt->hwPt());
	  isoObjEtaHistEmul->Fill(emulIt->hwEta());
	  isoObjPhiHistEmul->Fill(emulIt->hwPhi());
	} else {
	  objEtHistData->Fill(dataIt->hwPt());
	  objEtaHistData->Fill(dataIt->hwEta());
	  objPhiHistData->Fill(dataIt->hwPhi());

	  objEtHistEmul->Fill(emulIt->hwPt());
	  objEtaHistEmul->Fill(emulIt->hwEta());
	  objPhiHistEmul->Fill(emulIt->hwPhi());
	}
      }

      // if only position agrees
      if (posGood && !etGood) {
	if (iso) {
	  ++etOffIsoEGs;
	  objSummaryHist->Fill(ISOEGETOFF);
	} else {
	  ++etOffEGs;
	  objSummaryHist->Fill(EGETOFF);
	}
      }

      // if only energy agrees
      if (!posGood && etGood) {
	if (iso) {
	  ++posOffIsoEGs;
	  objSummaryHist->Fill(ISOEGPOSOFF);
	} else {
	  ++posOffEGs;
	  objSummaryHist->Fill(EGPOSOFF);
	}
      }

      // keep track of number of objects
      if (iso) {
	++totalIsoEGs;
      } else {
	++totalEGs;
      }

      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
	  emulIt == emulCol->end(currBx))
	break;
    }
  } else {
    return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for taus
bool L1TdeStage2CaloLayer2::compareTaus(
  const edm::Handle<l1t::TauBxCollection> & dataCol,
  const edm::Handle<l1t::TauBxCollection> & emulCol)
{
  bool eventGood = true;

  l1t::TauBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::TauBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  TH1F * isoObjEtHistData = isoTauEtData->getTH1F();
  TH1F * isoObjPhiHistData = isoTauPhiData->getTH1F();
  TH1F * isoObjEtaHistData = isoTauEtaData->getTH1F();
  TH1F * isoObjEtHistEmul = isoTauEtEmul->getTH1F();
  TH1F * isoObjPhiHistEmul = isoTauPhiEmul->getTH1F();
  TH1F * isoObjEtaHistEmul = isoTauEtaEmul->getTH1F();

  TH1F * objEtHistData = tauEtData->getTH1F();
  TH1F * objPhiHistData = tauPhiData->getTH1F();
  TH1F * objEtaHistData = tauEtaData->getTH1F();
  TH1F * objEtHistEmul = tauEtEmul->getTH1F();
  TH1F * objPhiHistEmul = tauPhiEmul->getTH1F();
  TH1F * objEtaHistEmul = tauEtaEmul->getTH1F();

  TH1F * objSummaryHist = tauSummary->getTH1F();
  TH1F * summaryHist = agreementSummary->getTH1F();

  // check length of collections
  if (dataCol->size() != emulCol->size()) {

    if (dataCol->size() < emulCol->size()) {
      if (dataCol->size() < 1)
	return false;

      // if there are more events in data loop over the data collection
      while (true) {

	// Populate different set of histograms if object is solated

	if (dataIt->hwIso()) {
	  isoObjEtHistData->Fill(dataIt->hwPt());
	  isoObjEtaHistData->Fill(dataIt->hwEta());
	  isoObjPhiHistData->Fill(dataIt->hwPhi());
	} else {
	  objEtHistData->Fill(dataIt->hwPt());
	  objEtaHistData->Fill(dataIt->hwEta());
	  objPhiHistData->Fill(dataIt->hwPhi());
	}

	++dataIt;

	if (dataIt == dataCol->end(currBx))
	  break;
      }
    } else {

      if (emulCol->size() < 1)
	return false;

      while (true) {

	// Populate different set of histograms if object is solated

	if(emulIt->hwIso()) {
	  isoObjEtHistEmul->Fill(emulIt->hwPt());
	  isoObjEtaHistEmul->Fill(emulIt->hwEta());
	  isoObjPhiHistEmul->Fill(emulIt->hwPhi());
	} else {
	  objEtHistEmul->Fill(emulIt->hwPt());
	  objEtaHistEmul->Fill(emulIt->hwEta());
	  objPhiHistEmul->Fill(emulIt->hwPhi());
	}

	++emulIt;

	if (emulIt == emulCol->end(currBx))
	  break;
      }
    }

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

      // object position mismatch
      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	posGood = false;
	eventGood = false;
      }
      if (dataIt->hwEta() != emulIt->hwEta()) {
	posGood = false;
	eventGood = false;
      }

      // if both position and energy agree, object is good
      if (eventGood) {
	summaryHist->Fill(TAUGOOD_S);

	if (iso) {
	  ++goodIsoTaus;
	  objSummaryHist->Fill(ISOTAUGOOD);
	} else {
	  ++goodTaus;
	  objSummaryHist->Fill(TAUGOOD);
	}
      } else {

	if (iso) {
	  isoObjEtHistData->Fill(dataIt->hwPt());
	  isoObjEtaHistData->Fill(dataIt->hwEta());
	  isoObjPhiHistData->Fill(dataIt->hwPhi());

	  isoObjEtHistEmul->Fill(emulIt->hwPt());
	  isoObjEtaHistEmul->Fill(emulIt->hwEta());
	  isoObjPhiHistEmul->Fill(emulIt->hwPhi());

	} else {
	  objEtHistData->Fill(dataIt->hwPt());
	  objEtaHistData->Fill(dataIt->hwEta());
	  objPhiHistData->Fill(dataIt->hwPhi());

	  objEtHistEmul->Fill(emulIt->hwPt());
	  objEtaHistEmul->Fill(emulIt->hwEta());
	  objPhiHistEmul->Fill(emulIt->hwPhi());
	}
      }

      // if only position agrees
      if (posGood && !etGood) {
	if (iso) {
	  ++etOffIsoTaus;
	  objSummaryHist->Fill(ISOTAUETOFF);
	} else {
	  ++etOffTaus;
	  objSummaryHist->Fill(TAUETOFF);
	}
      }

      // if only energy agrees
      if (!posGood && etGood) {
	if (iso) {
	  ++posOffIsoTaus;
	  objSummaryHist->Fill(ISOTAUPOSOFF);
	} else {
	  ++posOffTaus;
	  objSummaryHist->Fill(TAUPOSOFF);
	}
      }

      // keep track of number of objects
      if (iso) {
	++totalIsoTaus;
      } else {
	++totalTaus;
      }

      // increment position of pointers
      ++dataIt;
      ++emulIt;

      if (dataIt == dataCol->end(currBx) ||
	  emulIt == emulCol->end(currBx))
	break;
    }
  } else {
    return false;
  }

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

// comparison method for sums
bool L1TdeStage2CaloLayer2::compareSums(
  const edm::Handle<l1t::EtSumBxCollection> & dataCol,
  const edm::Handle<l1t::EtSumBxCollection> & emulCol)
{
  bool eventGood = true;

  bool etGood = true;
  bool phiGood = true;

  l1t::EtSumBxCollection::const_iterator dataIt = dataCol->begin(currBx);
  l1t::EtSumBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  TH1F * objSummaryHist = sumSummary->getTH1F();
  TH1F * summaryHist = agreementSummary->getTH1F();

  while(true) {

    // It should be possible to implement this with a switch statement
    etGood = true;
    phiGood = true;

    // ETT
    if (l1t::EtSum::EtSumType::kTotalEt == dataIt->getType()) {

      ++totalETTSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;

	ettData->getTH1F()->Fill(dataIt->hwPt());
	ettEmul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodETTSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(ETTSUMGOOD);
      }
    }

    // ETTHF
    if (l1t::EtSum::EtSumType::kTotalEtHF == dataIt->getType()) {

      ++totalETTSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;

	ettHFData->getTH1F()->Fill(dataIt->hwPt());
	ettHFEmul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodETTSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(ETTSUMGOOD);
      }
    }

    // ETTEM
    if (l1t::EtSum::EtSumType::kTotalEtEm == dataIt->getType()) {

      ++totalETTSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	ettEmData->getTH1F()->Fill(dataIt->hwPt());
	ettEmEmul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodETTSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(ETTSUMGOOD);
      }
    }

    // HTT
    if (l1t::EtSum::EtSumType::kTotalHt == dataIt->getType()) {

      ++totalHTTSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	httData->getTH1F()->Fill(dataIt->hwPt());
	httEmul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodHTTSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(HTTSUMGOOD);
      }
    }

    // HTTHF
    if (l1t::EtSum::EtSumType::kTotalHtHF == dataIt->getType()) {

      ++totalHTTSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	httHFData->getTH1F()->Fill(dataIt->hwPt());
	httHFEmul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodHTTSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(HTTSUMGOOD);
      }
    }

    // MET
    if (l1t::EtSum::EtSumType::kMissingEt == dataIt->getType()) {

      ++totalMETSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	etGood = false;
	eventGood = false;
      }

      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	phiGood = false;
	eventGood = false;
      }

      if (etGood && phiGood) {
	++goodSums;
	++goodMETSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(METSUMGOOD);
      } else {
	metEtData->getTH1F()->Fill(dataIt->hwPt());
	metPhiData->getTH1F()->Fill(dataIt->hwPhi());
	metEtEmul->getTH1F()->Fill(emulIt->hwPt());
	metPhiEmul->getTH1F()->Fill(emulIt->hwPhi());
      }
    }

    // METHF
    if (l1t::EtSum::EtSumType::kMissingEtHF == dataIt->getType()) {

      ++totalMETSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	etGood = false;
	eventGood = false;
      }

      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	phiGood = false;
	eventGood = false;
      }

      if (etGood && phiGood) {
	++goodSums;
	++goodMETSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(METSUMGOOD);
      } else {
	metHFEtData->getTH1F()->Fill(dataIt->hwPt());
	metHFPhiData->getTH1F()->Fill(dataIt->hwPhi());
	metHFEtEmul->getTH1F()->Fill(emulIt->hwPt());
	metHFPhiEmul->getTH1F()->Fill(emulIt->hwPhi());
      }
    }

    // MHT
    if (l1t::EtSum::EtSumType::kMissingHtHF == dataIt->getType()) {

      ++totalMHTSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	etGood = false;
	eventGood = false;
      }

      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	phiGood = false;
	eventGood = false;
      }

      if (etGood && phiGood) {
	++goodSums;
	++goodMHTSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(MHTSUMGOOD);
      } else {
	mhtEtData->getTH1F()->Fill(dataIt->hwPt());
	mhtPhiData->getTH1F()->Fill(dataIt->hwPhi());
	mhtEtEmul->getTH1F()->Fill(emulIt->hwPt());
	mhtPhiEmul->getTH1F()->Fill(emulIt->hwPhi());
      }
    }

    // MHTHF
    if (l1t::EtSum::EtSumType::kMissingEt == dataIt->getType()) {

      ++totalMHTSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	phiGood = false;
	eventGood = false;
      }

      if (dataIt->hwPhi() != emulIt->hwPhi()) {
	phiGood = false;
	eventGood = false;
      }

      if (etGood && phiGood) {
	++goodSums;
	++goodMHTSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(MHTSUMGOOD);
      } else {
       	mhtHFEtData->getTH1F()->Fill(dataIt->hwPt());
	mhtHFPhiData->getTH1F()->Fill(dataIt->hwPhi());
	mhtHFEtEmul->getTH1F()->Fill(emulIt->hwPt());
	mhtHFPhiEmul->getTH1F()->Fill(emulIt->hwPhi());
      }
    }

    // MBHFP0
    if (l1t::EtSum::EtSumType::kMinBiasHFP0 == dataIt->getType()) {

      ++totalMBHFSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	mbhfp0Data->getTH1F()->Fill(dataIt->hwPt());
	mbhfp0Emul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodMBHFSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(MBHFSUMGOOD);
      }
    }

    // MBHFM0
    if (l1t::EtSum::EtSumType::kMinBiasHFM0 == dataIt->getType()) {

      ++totalMBHFSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	mbhfm0Data->getTH1F()->Fill(dataIt->hwPt());
	mbhfm0Emul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodMBHFSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(MBHFSUMGOOD);
      }
    }

    // MBHFP1
    if (l1t::EtSum::EtSumType::kMinBiasHFP1 == dataIt->getType()) {

      ++totalMBHFSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	mbhfp1Data->getTH1F()->Fill(dataIt->hwPt());
	mbhfp1Emul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodMBHFSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(MBHFSUMGOOD);
      }
    }

    // MBHFM1
    if (l1t::EtSum::EtSumType::kMinBiasHFM1 == dataIt->getType()) {

      ++totalMBHFSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	mbhfm1Data->getTH1F()->Fill(dataIt->hwPt());
	mbhfm1Emul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodMBHFSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(MBHFSUMGOOD);
      }
    }

    // TowerCount
    if (l1t::EtSum::EtSumType::kTowerCount == dataIt->getType()) {

      ++totalTowCountSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
	towCountData->getTH1F()->Fill(dataIt->hwPt());
	towCountEmul->getTH1F()->Fill(emulIt->hwPt());
      } else {
	++goodSums;
	++goodTowCountSums;
	summaryHist->Fill(SUMGOOD_S);
	objSummaryHist->Fill(SUMGOOD);
	objSummaryHist->Fill(TOWCOUNTGOOD);
      }
    }

    ++totalSums;

    ++dataIt;
    ++emulIt;

    if (dataIt == dataCol->end(currBx) || emulIt == emulCol->end(currBx))
      break;
  }

  if (eventGood)
    summaryHist->Fill(SUMGOOD_S);

  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}

DEFINE_FWK_MODULE (L1TdeStage2CaloLayer2);
