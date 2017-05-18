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

L1TdeStage2CaloLayer2::~L1TdeStage2CaloLayer2()
{

}

void L1TdeStage2CaloLayer2::dqmBeginRun (
  edm::Run const &,
  edm::EventSetup const & evSetup) {}

void L1TdeStage2CaloLayer2::beginLuminosityBlock (
  const edm::LuminosityBlock& iLumi,
  const edm::EventSetup & evSetup) {}

void L1TdeStage2CaloLayer2::endLuminosityBlock(
  const edm::LuminosityBlock& lumi,
  const edm::EventSetup& evSetup)
{
  double totRatio = totalEvents != 0 ? goodEvents / (totalEvents * 1.0) : 0;
  double jetRatio = totalJets != 0 ? goodJets / (totalJets * 1.0) : 0;
  double egRatio = (totalEGs + totalIsoEGs) != 0 ?
    (goodEGs + goodIsoEGs) / ((totalEGs + totalIsoEGs) * 1.0) : 0;
  double tauRatio = (totalTaus + totalIsoTaus) != 0 ?
    (goodTaus + goodIsoTaus) / ((totalTaus + totalIsoTaus) * 1.0) : 0;
  double sumRatio = totalSums != 0 ? goodSums / (totalSums * 1.0) : 0;

  // divide the bin contents of high-level summary histogram by the totals
  // this might require too much memory (called every LS)
  TH1F * hlSum = agreementSummary->getTH1F();
  hlSum->SetBinContent(hlSum->FindBin(EVENTGOOD), totRatio);
  hlSum->SetBinContent(hlSum->FindBin(JETGOOD_S), jetRatio);
  // need to make a distinction between iso and non-iso eg and tau.
  // I feel that the variables used here are for non iso quantities only
  hlSum->SetBinContent(hlSum->FindBin(EGGOOD_S), egRatio);
  hlSum->SetBinContent(hlSum->FindBin(TAUGOOD_S), tauRatio);
  hlSum->SetBinContent(hlSum->FindBin(SUMGOOD_S), sumRatio);

  std::cout << "goodEvents totalEvents ratio: " << goodEvents << " "
	    << totalEvents << " " << totRatio << std::endl;
  std::cout << "goodJets totalJets ratio: " << goodJets << " " << totalJets
	    << " " << jetRatio << std::endl;
  std::cout << "goodEGs totalEG ratio: " << goodEGs << " " << totalEGs
	    << " " << egRatio << std::endl;
  std::cout << "goodIsoEGs totalIsoEG ratio: " << goodIsoEGs << " "
	    << totalIsoEGs << " " << egRatio << std::endl;
  std::cout << "goodTaus totalTaus ratio: " << goodTaus << " " << totalTaus
	    << " " << tauRatio << std::endl;
  std::cout << "goodSums totalSums ratio: " << goodSums << " " << totalSums
	    << " " << sumRatio << std::endl;

  // this might require too much memory (called every LS)
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

/**
   Method to declare or "book" all histograms that will be part of module

   Histograms that are to be visualised as part of the DQM module should be
   registered with the IBooker object any additional configuration such as title
   or axis labels and ranges. A good rule of thumb for the amount of
   configuration is that it should be possible to understnand the contents of
   the histogram using the configuration received from this method since the
   plots generated by this module would later be stored into ROOT files for
   transfer to the DQM system and it should be possible to ...

   @param DQMStore::IBooker&      ibooker
   @param edm::Run const &
   @param edm::EventSetup const &

   @return void
 */
void L1TdeStage2CaloLayer2::bookHistograms(
  DQMStore::IBooker &ibooker,
  edm::Run const &,
  edm::EventSetup const&) {

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

  ibooker.setCurrentFolder(monitorDir + "/Problematic Sums");
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

  mpSummary = ibooker.book1D("MP Specific Bad Events Summary", "MP Summary",
			     8, 1, 9);
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

  // std::cout << "Jet event by event comparisons. " << std::endl;

  // unsigned int currBx = 0;
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


  l1t::EGammaBxCollection::const_iterator dataEGIt = egDataCol->begin(currBx);
  l1t::EGammaBxCollection::const_iterator emulEGIt = egEmulCol->begin(currBx);

  l1t::TauBxCollection::const_iterator dataTauIt = tauDataCol->begin(currBx);
  l1t::TauBxCollection::const_iterator emulTauIt = tauEmulCol->begin(currBx);

  l1t::EtSumBxCollection::const_iterator dataSumIt = sumDataCol->begin(currBx);
  l1t::EtSumBxCollection::const_iterator emulSumIt = sumEmulCol->begin(currBx);

  TH1F * hist = agreementSummary->getTH1F();

  // TH1F * jetHist = jetSummary->getTH1F();
  // TH1F * jetEtHist = jetEt->getTH1F();
  // TH1F * jetEtaHist = jetEta->getTH1F();
  // TH1F * jetPhiHist = jetPhi->getTH1F();

  // TH1F * egHist = egSummary->getTH1F();
  // TH1F * egEtHist = egEt->getTH1F();
  // TH1F * egEtaHist = egEta->getTH1F();
  // TH1F * egPhiHist = egPhi->getTH1F();
  // TH1F * isoEgEtHist = isoEgEt->getTH1F();
  // TH1F * isoEgEtaHist = isoEgEta->getTH1F();
  // TH1F * isoEgPhiHist = isoEgPhi->getTH1F();

  // TH1F * tauHist = tauSummary->getTH1F();
  // TH1F * tauEtHist = tauEt->getTH1F();
  // TH1F * tauEtaHist = tauEta->getTH1F();
  // TH1F * tauPhiHist = tauPhi->getTH1F();
  // TH1F * isoTauEtHist = isoTauEt->getTH1F();
  // TH1F * isoTauEtaHist = isoTauEta->getTH1F();
  // TH1F * isoTauPhiHist = isoTauPhi->getTH1F();

  // TH1F * sumHist = sumSummary->getTH1F();

  if (!compareJets(jetDataCol, jetEmulCol))
    eventGood = false;

  if (!compareEGs(egDataCol, egEmulCol))
    eventGood = false;

  if (!compareTaus(tauDataCol, tauEmulCol))
    eventGood = false;

  if (!compareSums(sumDataCol, sumEmulCol))
    eventGood = false;

  // loop over the different bx associated with the collections (choose one)

  // at each iteration:

  // skip BXs which do not exist in the "other" collection (the one not being
  // looped over)

  // extract the data and emul jet collections for each BX
  // this loop can be used to populate all histograms associated with a given
  // object, i.e. pT, eta, phi, etc

  // while looping over the two collections one can assume that both collections
  // have the objects sorted in the same order which would only require to
  // compare the objects available at the current iteration.

  /**
     Questions:
     - what could make the data and emul bx ranges to be different?
     - how can I confirm that the emulator data is being filled?
  */

  if (eventGood) {
    hist->Fill(EVENTGOOD);
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
      if (dataCol->size() < 1)
	return false;

      while (true) {
	objEtHistData->Fill(dataIt->hwPt());
	objEtaHistData->Fill(dataIt->hwEta());
	objPhiHistData->Fill(dataIt->hwPhi());

	++dataIt;

	if (dataIt == dataCol->end(currBx))
	  break;
      }
    } else{

      if (emulCol->size() < 1)
	return false;

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

      std::cout << "more EGs in data than emul" << std::endl;

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

      std::cout << "more EGs in emul than data"
		<< emulCol->size() << " " << dataCol->size() << std::endl;

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
      }
    }

    // MBHFP0
    if (l1t::EtSum::EtSumType::kMinBiasHFP0 == dataIt->getType()) {

      ++totalMBHFSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
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
	++goodMBHFSums;
      if (dataIt->hwPt() != emulIt->hwPt()) {
	eventGood = false;
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
