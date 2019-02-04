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
    verbose(ps.getUntrackedParameter<bool> ("verbose", false)),
    enable2DComp(ps.getUntrackedParameter<bool> ("enable2DComp", false)) // When true eta-phi comparison plots are also produced
{}

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
  //if enable2DComp is true book also 2D eta-phi plots
  if(enable2DComp){
    jet2DEtaPhiData = ibooker.book2D("Problematic Data Jet Eta - Phi","Jet #eta - #phi map",
                                      50,-5.,5.,25,-3.2,3.2);
    jet2DEtaPhiData->setAxisTitle("#eta", 1);
    jet2DEtaPhiData->setAxisTitle("#phi", 2);

    jet2DEtaPhiEmul = ibooker.book2D("Problematic Emul Jet Eta - Phi","Jet #eta - #phi map",
                                      50,-5.,5.,25,-3.2,3.2);
    jet2DEtaPhiEmul->setAxisTitle("#eta", 1);
    jet2DEtaPhiEmul->setAxisTitle("#phi", 2);
  }

  // DQM directory to store histograms with problematic e/gs
  ibooker.setCurrentFolder(monitorDir + "/Problematic EG candidates");

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
  //if enable2DComp is true book also 2D eta-phi plots
  if(enable2DComp){
    eg2DEtaPhiData = ibooker.book2D("Problematic Data Eg Eta - Phi","Eg #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    eg2DEtaPhiData->setAxisTitle("#eta", 1);
    eg2DEtaPhiData->setAxisTitle("#phi", 2);

    eg2DEtaPhiEmul = ibooker.book2D("Problematic Emul Eg Eta - Phi","Eg #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    eg2DEtaPhiEmul->setAxisTitle("#eta", 1);
    eg2DEtaPhiEmul->setAxisTitle("#phi", 2);
  }
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
  // enable2DComp is true book also 2D eta-phi plots
  if(enable2DComp){
    isoEg2DEtaPhiData = ibooker.book2D("Problematic Isolated Data Eg Eta - Phi","Iso Eg #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    isoEg2DEtaPhiData->setAxisTitle("#eta", 1);
    isoEg2DEtaPhiData->setAxisTitle("#phi", 2);

    isoEg2DEtaPhiEmul = ibooker.book2D("Problematic Isolated Emul Eg Eta - Phi","Iso Eg #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    isoEg2DEtaPhiEmul->setAxisTitle("#eta", 1);
    isoEg2DEtaPhiEmul->setAxisTitle("#phi", 2);
  }
  // DQM directory to store histograms with problematic taus
  ibooker.setCurrentFolder(monitorDir + "/Problematic Tau candidates");

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
  // enable2DComp is true book also 2D eta-phi plots
  if(enable2DComp){
    tau2DEtaPhiData = ibooker.book2D("Problematic Data Tau Eta - Phi","Tau #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    tau2DEtaPhiData->setAxisTitle("#eta", 1);
    tau2DEtaPhiData->setAxisTitle("#phi", 2);

    tau2DEtaPhiEmul = ibooker.book2D("Problematic Emul Tau Eta - Phi","Tau #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    tau2DEtaPhiEmul->setAxisTitle("#eta", 1);
    tau2DEtaPhiEmul->setAxisTitle("#phi", 2);
  }
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
  // enable2DComp is true book also 2D eta-phi plots
  if(enable2DComp){
    isoTau2DEtaPhiData = ibooker.book2D("Problematic Isolated Data Tau Eta - Phi","Iso Tau #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    isoTau2DEtaPhiData->setAxisTitle("#eta", 1);
    isoTau2DEtaPhiData->setAxisTitle("#phi", 2);

    isoTau2DEtaPhiEmul = ibooker.book2D("Problematic Isolated Emul Tau Eta - Phi","Iso Tau #eta - #phi map",
                                      30,-3.,3.,25,-3.2,3.2);
    isoTau2DEtaPhiEmul->setAxisTitle("#eta", 1);
    isoTau2DEtaPhiEmul->setAxisTitle("#phi", 2);
  }
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

 // book asymmetry count
 
 asymCountData = ibooker.book1D("Problematic Asymmetry Count - Data",
                                "", 256, -0.5, 255.5);
 asymCountEmul = ibooker.book1D("Problematic Asymmetry Count - Emulator",
                                "", 256, -0.5, 255.5);

 // book centrality sums
 
 centrCountData = ibooker.book1D("Problematic Centrality Count - Data",
                                "", 9, -1.5, 7.5);
 centrCountEmul = ibooker.book1D("Problematic Centrality Count - Emulator",
                                "", 9, -1.5, 7.5);




  // setup the directory where the histograms are to be visualised, value is set
  // in constructor and taken from python configuration file for module
  ibooker.setCurrentFolder(monitorDir + "/expert");


  // Jet energy in MP firmware is stored in 16 bits which sets the range of
  // jet energy to 2^16 * 0.5 GeV = 32768 GeV (65536 hardware units)
  // --- this is only for MP jets, the demux jets have much decreased precision
  // --- and this should be replaced

  // the index of the first bin in histogram should match value of first enum
  agreementSummary = ibooker.book1D(
    "CaloL2 Object Agreement Summary",
    "CaloL2 event-by-event object agreement fractions", 10, 1, 11);

  agreementSummary->setBinLabel(EVENTGOOD, "good events");
  agreementSummary->setBinLabel(NEVENTS, "total events");
  agreementSummary->setBinLabel(NJETS_S, "total jets");
  agreementSummary->setBinLabel(JETGOOD_S, "good jets");
  agreementSummary->setBinLabel(NEGS_S, "total e/gs");
  agreementSummary->setBinLabel(EGGOOD_S, "good e/gs");
  agreementSummary->setBinLabel(NTAUS_S, "total taus");
  agreementSummary->setBinLabel(TAUGOOD_S, "good taus");
  agreementSummary->setBinLabel(NSUMS_S, "total sums");
  agreementSummary->setBinLabel(SUMGOOD_S, "good sums");

  jetSummary = ibooker.book1D(
    "Jet Agreement Summary", "Jet Agreement Summary", 4, 1, 5);
  jetSummary->setBinLabel(NJETS, "total jets");
  jetSummary->setBinLabel(JETGOOD, "good jets");
  jetSummary->setBinLabel(JETPOSOFF, "jets pos off only");
  jetSummary->setBinLabel(JETETOFF, "jets Et off only ");

  egSummary = ibooker.book1D(
    "EG Agreement Summary", "EG Agreement Summary", 8, 1, 9);
  egSummary->setBinLabel(NEGS, "total non-iso e/gs");
  egSummary->setBinLabel(EGGOOD, "good non-iso e/gs");
  egSummary->setBinLabel(EGPOSOFF, "non-iso e/gs pos off");
  egSummary->setBinLabel(EGETOFF, "non-iso e/gs Et off");
  egSummary->setBinLabel(NISOEGS, "total iso e/gs");
  egSummary->setBinLabel(ISOEGGOOD, "good iso e/gs");
  egSummary->setBinLabel(ISOEGPOSOFF, "iso e/gs pos off");
  egSummary->setBinLabel(ISOEGETOFF, "iso e/gs Et off");

  tauSummary = ibooker.book1D(
    "Tau Agreement Summary", "Tau Agreement Summary", 8, 1, 9);
  tauSummary->setBinLabel(NTAUS, "total taus");
  tauSummary->setBinLabel(TAUGOOD, "good non-iso taus");
  tauSummary->setBinLabel(TAUPOSOFF, "non-iso taus pos off");
  tauSummary->setBinLabel(TAUETOFF, "non-iso taus Et off");
  tauSummary->setBinLabel(NISOTAUS, "total iso taus");
  tauSummary->setBinLabel(ISOTAUGOOD, "good iso taus");
  tauSummary->setBinLabel(ISOTAUPOSOFF, "iso taus pos off");
  tauSummary->setBinLabel(ISOTAUETOFF, "iso taus Et off");

  sumSummary = ibooker.book1D(
    "Energy Sum Agreement Summary", "Sum Agreement Summary", 18, 1, 19);
  sumSummary->setBinLabel(NSUMS, "total sums");
  sumSummary->setBinLabel(SUMGOOD, "good sums");
  sumSummary->setBinLabel(NETTSUMS, "total ETT sums");
  sumSummary->setBinLabel(ETTSUMGOOD, "good ETT sums");
  sumSummary->setBinLabel(NHTTSUMS, "total HTT sums");
  sumSummary->setBinLabel(HTTSUMGOOD, "good HTT sums");
  sumSummary->setBinLabel(NMETSUMS, "total MET sums");
  sumSummary->setBinLabel(METSUMGOOD, "good MET sums");
  sumSummary->setBinLabel(NMHTSUMS, "total MHT sums");
  sumSummary->setBinLabel(MHTSUMGOOD, "good MHT sums");
  sumSummary->setBinLabel(NMBHFSUMS, "total MBHF sums");
  sumSummary->setBinLabel(MBHFSUMGOOD, "good MBHF sums");
  sumSummary->setBinLabel(NTOWCOUNTS, "total TowCount sums");
  sumSummary->setBinLabel(TOWCOUNTGOOD, "good TowCount sums");
  sumSummary->setBinLabel(NASYMCOUNTS, "total AsymCount sums");
  sumSummary->setBinLabel(ASYMCOUNTGOOD, "good AsymCount sums");
  sumSummary->setBinLabel(NCENTRCOUNTS, "total CentrCount sums");
  sumSummary->setBinLabel(CENTRCOUNTGOOD, "good CentrCount sums");
  

  // high level directory
  ibooker.setCurrentFolder(monitorDir);

  problemSummary = ibooker.book1D(
    "Problem Summary", "Problematic Event Summary", 8, 1, 9);
  problemSummary->setBinLabel(NEVENTS_P, "total events");
  problemSummary->setBinLabel(JETCOLLSIZE, "jet collection size");
  problemSummary->setBinLabel(EGCOLLSIZE, "eg collection size");
  problemSummary->setBinLabel(TAUCOLLSIZE, "tau collection size");
  problemSummary->setBinLabel(JETMISMATCH, "jet mismatch");
  problemSummary->setBinLabel(EGMISMATCH, "eg mismatch");
  problemSummary->setBinLabel(TAUMISMATCH, "tau mismatch");
  problemSummary->setBinLabel(SUMMISMATCH, "sum mismatch");
}
void L1TdeStage2CaloLayer2::analyze (
  const edm::Event& e,
  const edm::EventSetup & c) {

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

  if (!compareJets(jetDataCol, jetEmulCol)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: jet problem "
					    << std::endl;
    problemSummary->Fill(JETMISMATCH);
    eventGood = false;
  }

  if (!compareEGs(egDataCol, egEmulCol)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: eg problem "
					    << std::endl;
    problemSummary->Fill(EGMISMATCH);
    eventGood = false;
  }

  if (!compareTaus(tauDataCol, tauEmulCol)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: tau problem "
					    << std::endl;
    problemSummary->Fill(TAUMISMATCH);
    eventGood = false;
  }

  if (!compareSums(sumDataCol, sumEmulCol)) {
    if (verbose)
      edm::LogInfo("L1TdeStage2CaloLayer2") << "l1t calol2 dqm: sum problem "
					    << std::endl;
    problemSummary->Fill(SUMMISMATCH);
    eventGood = false;
  }

  /**
     Questions:
     - what could make the data and emul bx ranges to be different?
     - how can I confirm that the emulator data is being filled?
  */

  if (eventGood) {
    agreementSummary->Fill(EVENTGOOD);
  }

  agreementSummary->Fill(NEVENTS);
  problemSummary->Fill(NEVENTS_P);
}

// comparison method for jets
bool L1TdeStage2CaloLayer2::compareJets(
  const edm::Handle<l1t::JetBxCollection> & dataCol,
  const edm::Handle<l1t::JetBxCollection> & emulCol)
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
	jetEtData->Fill(dataIt->hwPt());
	jetEtaData->Fill(dataIt->hwEta());
	jetPhiData->Fill(dataIt->hwPhi());
        if(enable2DComp) jet2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());

	++dataIt;

	if (dataIt == dataCol->end(currBx))
	  break;
      }
    } else {

      // if only the emul collection is empty, declare event as bad
      if (emulCol->isEmpty(currBx)) return false;

      while (true) {

	jetEtEmul->Fill(emulIt->hwPt());
	jetEtaEmul->Fill(emulIt->hwEta());
	jetPhiEmul->Fill(emulIt->hwPhi()); 
        if(enable2DComp) jet2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());

	++emulIt;

	if (emulIt == emulCol->end(currBx))
	  break;
      }

    }

    problemSummary->Fill(JETCOLLSIZE);
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
	agreementSummary->Fill(JETGOOD_S);
	jetSummary->Fill(JETGOOD);
      } else {
	jetEtData->Fill(dataIt->hwPt());
	jetEtaData->Fill(dataIt->hwEta());
	jetPhiData->Fill(dataIt->hwPhi());
        if(enable2DComp) jet2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());

	jetEtEmul->Fill(emulIt->hwPt());
	jetEtaEmul->Fill(emulIt->hwEta());
	jetPhiEmul->Fill(emulIt->hwPhi());
        if(enable2DComp) jet2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());

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
	jetSummary->Fill(JETETOFF);
      }

      // if only energy agrees
      if (!posGood && etGood) {
	jetSummary->Fill(JETPOSOFF);
      }

      // keep track of jets
      agreementSummary->Fill(NJETS_S);
      jetSummary->Fill(NJETS);

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
  const edm::Handle<l1t::EGammaBxCollection> & emulCol)
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
	  isoEgEtData->Fill(dataIt->hwPt());
	  isoEgEtaData->Fill(dataIt->hwEta());
	  isoEgPhiData->Fill(dataIt->hwPhi());  
          if(enable2DComp) isoEg2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());
	} else {
	  egEtData->Fill(dataIt->hwPt());
	  egEtaData->Fill(dataIt->hwEta());
	  egPhiData->Fill(dataIt->hwPhi());
          if(enable2DComp) eg2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());
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
	  isoEgEtEmul->Fill(emulIt->hwPt());
	  isoEgEtaEmul->Fill(emulIt->hwEta());
	  isoEgPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) isoEg2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());    
	} else {
	  egEtEmul->Fill(emulIt->hwPt());
	  egEtaEmul->Fill(emulIt->hwEta());
	  egPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) eg2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi()); 
	}

	++emulIt;

	if (emulIt == emulCol->end(currBx))
	  break;
      }
    }

    problemSummary->Fill(EGCOLLSIZE);
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
	agreementSummary->Fill(EGGOOD_S);

	if (iso) {
	  egSummary->Fill(ISOEGGOOD);
	} else {
	  egSummary->Fill(EGGOOD);
	}

      } else {

	if (iso) {
	  isoEgEtData->Fill(dataIt->hwPt());
	  isoEgEtaData->Fill(dataIt->hwEta());
	  isoEgPhiData->Fill(dataIt->hwPhi());
          if(enable2DComp) isoEg2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());

	  isoEgEtEmul->Fill(emulIt->hwPt());
	  isoEgEtaEmul->Fill(emulIt->hwEta());
	  isoEgPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) isoEg2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());
	} else {
	  egEtData->Fill(dataIt->hwPt());
	  egEtaData->Fill(dataIt->hwEta());
	  egPhiData->Fill(dataIt->hwPhi());
          if(enable2DComp) eg2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());

	  egEtEmul->Fill(emulIt->hwPt());
	  egEtaEmul->Fill(emulIt->hwEta());
	  egPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) eg2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());
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
	  egSummary->Fill(ISOEGETOFF);
	} else {
	  egSummary->Fill(EGETOFF);
	}
      }

      // if only energy agrees
      if (!posGood && etGood) {
	if (iso) {
	  egSummary->Fill(ISOEGPOSOFF);
	} else {
	  egSummary->Fill(EGPOSOFF);
	}
      }

      // keep track of number of objects
      if (iso) {
	egSummary->Fill(NISOEGS);
      } else {
	egSummary->Fill(NEGS);
      }
      agreementSummary->Fill(NEGS_S);

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
  const edm::Handle<l1t::TauBxCollection> & emulCol)
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
	  isoTauEtData->Fill(dataIt->hwPt());
	  isoTauEtaData->Fill(dataIt->hwEta());
	  isoTauPhiData->Fill(dataIt->hwPhi()); 
          if(enable2DComp) isoTau2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());
	} else {
	  tauEtData->Fill(dataIt->hwPt());
	  tauEtaData->Fill(dataIt->hwEta());
	  tauPhiData->Fill(dataIt->hwPhi());
          if(enable2DComp) tau2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());
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
	  isoTauEtEmul->Fill(emulIt->hwPt());
	  isoTauEtaEmul->Fill(emulIt->hwEta());
	  isoTauPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) isoTau2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());
          
	} else {
	  tauEtEmul->Fill(emulIt->hwPt());
	  tauEtaEmul->Fill(emulIt->hwEta());
	  tauPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) tau2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());
	}

	++emulIt;

	if (emulIt == emulCol->end(currBx))
	  break;
      }
    }

    problemSummary->Fill(TAUCOLLSIZE);
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
	agreementSummary->Fill(TAUGOOD_S);

	if (iso) {
	  tauSummary->Fill(ISOTAUGOOD);
	} else {
	  tauSummary->Fill(TAUGOOD);
	}
      } else {

	if (iso) {
	  isoTauEtData->Fill(dataIt->hwPt());
	  isoTauEtaData->Fill(dataIt->hwEta());
	  isoTauPhiData->Fill(dataIt->hwPhi());
          if(enable2DComp) isoTau2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());

	  isoTauEtEmul->Fill(emulIt->hwPt());
	  isoTauEtaEmul->Fill(emulIt->hwEta());
	  isoTauPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) isoTau2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());

	} else {
	  tauEtData->Fill(dataIt->hwPt());
	  tauEtaData->Fill(dataIt->hwEta());
	  tauPhiData->Fill(dataIt->hwPhi());
          if(enable2DComp) tau2DEtaPhiData->Fill(dataIt->eta(), dataIt->phi());

	  tauEtEmul->Fill(emulIt->hwPt());
	  tauEtaEmul->Fill(emulIt->hwEta());
	  tauPhiEmul->Fill(emulIt->hwPhi());
          if(enable2DComp) tau2DEtaPhiEmul->Fill(emulIt->eta(), emulIt->phi());
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
	  tauSummary->Fill(ISOTAUETOFF);
	} else {
	  tauSummary->Fill(TAUETOFF);
	}
      }

      // if only energy agrees
      if (!posGood && etGood) {
	if (iso) {
	  tauSummary->Fill(ISOTAUPOSOFF);
	} else {
	  tauSummary->Fill(TAUPOSOFF);
	}
      }

      // keep track of number of objects
      if (iso) {
	tauSummary->Fill(NISOTAUS);
      } else {
	tauSummary->Fill(NTAUS);
      }

      agreementSummary->Fill(NTAUS_S);

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
  const edm::Handle<l1t::EtSumBxCollection> & emulCol)
{
  bool eventGood = true;

  bool etGood = true;
  bool phiGood = true;

  double dataEt = 0;
  double emulEt = 0;
  double dataPhi = 0;
  double emulPhi = 0;
  int dataCentr = 0;
  int emulCentr = 0;

  l1t::EtSumBxCollection::const_iterator dataIt;
  l1t::EtSumBxCollection::const_iterator emulIt = emulCol->begin(currBx);

  // if either data or emulator collections are empty mark the event as bad (this should never occur in normal running)
  // matching data/emul collections by type before proceeding with the checks
  if (dataCol->isEmpty(currBx) || emulCol->isEmpty(currBx))
    return false;
  
  while(true) {
    dataIt = dataCol->begin(currBx);
    while(true) {
      if (dataIt->getType()==emulIt->getType()){
        // It should be possible to implement this with a switch statement
        etGood = true;
        phiGood = true;
        // ETT
        if (l1t::EtSum::EtSumType::kTotalEt == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;

            ettData->Fill(dataEt);
            ettEmul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(ETTSUMGOOD);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NETTSUMS);
          sumSummary->Fill(NSUMS);
        }

        // ETTEM
        if (l1t::EtSum::EtSumType::kTotalEtEm == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;
            ettEmData->Fill(dataEt);
            ettEmEmul->Fill(emulEt);

          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(ETTSUMGOOD);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NETTSUMS);
          sumSummary->Fill(NSUMS);
        }

        // HTT
        if (l1t::EtSum::EtSumType::kTotalHt == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;
            httData->Fill(dataEt);
            httEmul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(HTTSUMGOOD);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NHTTSUMS);
          sumSummary->Fill(NSUMS);
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
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(METSUMGOOD);
          } else {
            metEtData->Fill(dataEt);
            metPhiData->Fill(dataPhi);
            metEtEmul->Fill(emulEt);
            metPhiEmul->Fill(emulPhi);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NMETSUMS);
          sumSummary->Fill(NSUMS);
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
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(METSUMGOOD);
          } else {
            metHFEtData->Fill(dataEt);
            metHFPhiData->Fill(dataPhi);
            metHFEtEmul->Fill(emulEt);
            metHFPhiEmul->Fill(emulPhi);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NMETSUMS);
          sumSummary->Fill(NSUMS);
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
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(MHTSUMGOOD);
          } else {
            mhtEtData->Fill(dataEt);
            mhtPhiData->Fill(dataPhi);
            mhtEtEmul->Fill(emulEt);
            mhtPhiEmul->Fill(emulPhi);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NMHTSUMS);
          sumSummary->Fill(NSUMS);
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
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(MHTSUMGOOD);
          } else {
            mhtHFEtData->Fill(dataEt);
            mhtHFPhiData->Fill(dataPhi);
            mhtHFEtEmul->Fill(emulEt);
            mhtHFPhiEmul->Fill(emulPhi);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NMHTSUMS);
          sumSummary->Fill(NSUMS);
        }

        // MBHFP0
        if (l1t::EtSum::EtSumType::kMinBiasHFP0 == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;
            mbhfp0Data->Fill(dataEt);
            mbhfp0Emul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(MBHFSUMGOOD);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NMBHFSUMS);
          sumSummary->Fill(NSUMS);
        }

        // MBHFM0
        if (l1t::EtSum::EtSumType::kMinBiasHFM0 == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;
            mbhfm0Data->Fill(dataEt);
            mbhfm0Emul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(MBHFSUMGOOD);
          }

          // update sum counters
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NMBHFSUMS);
          sumSummary->Fill(NSUMS);
        }

        // MBHFP1
        if (l1t::EtSum::EtSumType::kMinBiasHFP1 == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;
            mbhfp1Data->Fill(dataEt);
            mbhfp1Emul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(MBHFSUMGOOD);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NMBHFSUMS);
          sumSummary->Fill(NSUMS);
        }

        // MBHFM1
        if (l1t::EtSum::EtSumType::kMinBiasHFM1 == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          sumSummary->Fill(NMBHFSUMS);

          if (dataEt != emulEt) {
            eventGood = false;
            mbhfm1Data->Fill(dataEt);
            mbhfm1Emul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(MBHFSUMGOOD);
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

          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NSUMS);
        }

        // TowerCount
        if (l1t::EtSum::EtSumType::kTowerCount == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;
            towCountData->Fill(dataEt);
            towCountEmul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(TOWCOUNTGOOD);
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
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NTOWCOUNTS);
          sumSummary->Fill(NSUMS);
        }

        // AsymmetryCount
        if (l1t::EtSum::EtSumType::kAsymEt == dataIt->getType()) {

          dataEt = dataIt->hwPt();
          emulEt = emulIt->hwPt();

          if (dataEt != emulEt) {
            eventGood = false;
            asymCountData->Fill(dataEt);
            asymCountEmul->Fill(emulEt);
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(ASYMCOUNTGOOD);
          }

          if (verbose) {
            edm::LogInfo("L1TdeStage2CaloLayer2") << "AsymCount  | ";
            if (dataEt != emulEt)
              edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
            else
              edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";

            edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
            edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
          }
          // update sum counters
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NASYMCOUNTS);
          sumSummary->Fill(NSUMS);
        }

        // CentralityCount
        if (l1t::EtSum::EtSumType::kCentrality == dataIt->getType()) {
          dataCentr = dataIt->hwPt();
          emulCentr = emulIt->hwPt();
                      
          if (dataCentr != emulCentr) {
            eventGood = false;
            if (dataCentr==0) centrCountData->Fill(-1);
              else {
                for (int i=0; i<8; i++)
                  if (((dataCentr>>i)&1)==1) centrCountData->Fill(i);
            }
              
            if (emulCentr==0) centrCountEmul->Fill(-1);
              else {
                for (int i=0; i<8; i++)
                  if (((emulCentr>>i)&1)==1) centrCountEmul->Fill(i);
            }
              
          } else {
            agreementSummary->Fill(SUMGOOD_S);
            sumSummary->Fill(SUMGOOD);
            sumSummary->Fill(CENTRCOUNTGOOD);
          }

          if (verbose) {
            edm::LogInfo("L1TdeStage2CaloLayer2") << "CentrCount  | ";
            if (dataEt != emulEt)
              edm::LogInfo("L1TdeStage2CaloLayer2") << "x ";
            else
              edm::LogInfo("L1TdeStage2CaloLayer2") << "  ";

            edm::LogInfo("L1TdeStage2CaloLayer2") << dataEt << "\t" << emulEt;
            edm::LogInfo("L1TdeStage2CaloLayer2") << std::endl;
          }
          // update sum counters
          agreementSummary->Fill(NSUMS_S);
          sumSummary->Fill(NCENTRCOUNTS);
          sumSummary->Fill(NSUMS);
        }

        break;
      }
      ++dataIt;
      if (dataIt == dataCol->end(currBx))
      break;
		  
    }
	  
    ++emulIt;
    if (emulIt == emulCol->end(currBx))
      break;
  }
  // return a boolean that states whether the jet data in the event is in
  // agreement
  return eventGood;
}
