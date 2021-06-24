/**
 * \class L1TGT
 *
 *
 * Description: DQM for L1 Global Trigger.
 *
 * \author J. Berryhill, I. Mikulec
 * \author Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

#include "DQM/L1TMonitor/interface/L1TGT.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

L1TGT::L1TGT(const edm::ParameterSet& ps)
    : gtSource_L1GT_(consumes<L1GlobalTriggerReadoutRecord>(ps.getParameter<edm::InputTag>("gtSource"))),
      gtSource_L1MuGMT_(consumes<L1MuGMTReadoutCollection>(ps.getParameter<edm::InputTag>("gtSource"))),
      gtEvmSource_(consumes<L1GlobalTriggerEvmReadoutRecord>(ps.getParameter<edm::InputTag>("gtEvmSource"))),
      m_runInEventLoop(ps.getUntrackedParameter<bool>("runInEventLoop", false)),
      m_runInEndLumi(ps.getUntrackedParameter<bool>("runInEndLumi", false)),
      verbose_(ps.getUntrackedParameter<bool>("verbose", false)),
      m_nrEvJob(0),
      m_nrEvRun(0),
      preGps_(0ULL),
      preOrb_(0ULL) {
  m_histFolder = ps.getUntrackedParameter<std::string>("HistFolder", "L1T/L1TGT");
  l1gtTrigmenuToken_ = esConsumes<edm::Transition::BeginRun>();
}

L1TGT::~L1TGT() {
  // empty
}

void L1TGT::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& evSetup) {
  ibooker.setCurrentFolder(m_histFolder);

  // book histograms
  const int TotalNrBinsLs = 1000;
  const double totalNrBinsLs = static_cast<double>(TotalNrBinsLs);

  ibooker.setCurrentFolder(m_histFolder);

  algo_bits = ibooker.book1D("algo_bits", "GT algorithm trigger bits", 128, -0.5, 127.5);
  algo_bits->setAxisTitle("Algorithm trigger bits", 1);

  algo_bits_corr =
      ibooker.book2D("algo_bits_corr", "GT algorithm trigger bit correlation", 128, -0.5, 127.5, 128, -0.5, 127.5);
  algo_bits_corr->setAxisTitle("Algorithm trigger bits", 1);
  algo_bits_corr->setAxisTitle("Algorithm trigger bits", 2);

  tt_bits = ibooker.book1D("tt_bits", "GT technical trigger bits", 64, -0.5, 63.5);
  tt_bits->setAxisTitle("Technical trigger bits", 1);

  tt_bits_corr = ibooker.book2D("tt_bits_corr", "GT technical trigger bit correlation", 64, -0.5, 63.5, 64, -0.5, 63.5);
  tt_bits_corr->setAxisTitle("Technical trigger bits", 1);
  tt_bits_corr->setAxisTitle("Technical trigger bits", 2);

  algo_tt_bits_corr = ibooker.book2D(
      "algo_tt_bits_corr", "GT algorithm - technical trigger bit correlation", 128, -0.5, 127.5, 64, -0.5, 63.5);
  algo_tt_bits_corr->setAxisTitle("Algorithm trigger bits", 1);
  algo_tt_bits_corr->setAxisTitle("Technical trigger bits", 2);

  algo_bits_lumi = ibooker.book2D(
      "algo_bits_lumi", "GT algorithm trigger bit rate per LS", TotalNrBinsLs, 0., totalNrBinsLs, 128, -0.5, 127.5);
  algo_bits_lumi->setAxisTitle("Luminosity segment", 1);
  algo_bits_lumi->setAxisTitle("Algorithm trigger bits", 2);

  tt_bits_lumi = ibooker.book2D(
      "tt_bits_lumi", "GT technical trigger bit rate per LS", TotalNrBinsLs, 0., totalNrBinsLs, 64, -0.5, 63.5);
  tt_bits_lumi->setAxisTitle("Luminosity segment", 1);
  tt_bits_lumi->setAxisTitle("Technical trigger bits", 2);

  event_type = ibooker.book1D("event_type", "GT event type", 10, -0.5, 9.5);
  event_type->setAxisTitle("Event type", 1);
  event_type->setBinLabel(2, "Physics", 1);
  event_type->setBinLabel(3, "Calibration", 1);
  event_type->setBinLabel(4, "Random", 1);
  event_type->setBinLabel(6, "Traced", 1);
  event_type->setBinLabel(7, "Test", 1);
  event_type->setBinLabel(8, "Error", 1);

  event_number = ibooker.book1D("event_number", "GT event number (from last resync)", 100, 0., 50000.);
  event_number->setAxisTitle("Event number", 1);

  event_lumi = ibooker.bookProfile(
      "event_lumi", "GT event number (from last resync) vs LS", TotalNrBinsLs, 0., totalNrBinsLs, 100, -0.1, 1.e15, "s");
  event_lumi->setAxisTitle("Luminosity segment", 1);
  event_lumi->setAxisTitle("Event number", 2);

  trigger_number = ibooker.book1D("trigger_number", "GT trigger number (from start run)", 100, 0., 50000.);
  trigger_number->setAxisTitle("Trigger number", 1);

  trigger_lumi = ibooker.bookProfile("trigger_lumi",
                                     "GT trigger number (from start run) vs LS",
                                     TotalNrBinsLs,
                                     0.,
                                     totalNrBinsLs,
                                     100,
                                     -0.1,
                                     1.e15,
                                     "s");
  trigger_lumi->setAxisTitle("Luminosity segment", 1);
  trigger_lumi->setAxisTitle("Trigger number", 2);

  evnum_trignum_lumi = ibooker.bookProfile(
      "evnum_trignum_lumi", "GT event/trigger number ratio vs LS", TotalNrBinsLs, 0., totalNrBinsLs, 100, -0.1, 2., "s");
  evnum_trignum_lumi->setAxisTitle("Luminosity segment", 1);
  evnum_trignum_lumi->setAxisTitle("Event/trigger number ratio", 2);

  orbit_lumi = ibooker.bookProfile(
      "orbit_lumi", "GT orbit number vs LS", TotalNrBinsLs, 0., totalNrBinsLs, 100, -0.1, 1.e15, "s");
  orbit_lumi->setAxisTitle("Luminosity segment", 1);
  orbit_lumi->setAxisTitle("Orbit number", 2);

  setupversion_lumi = ibooker.bookProfile(
      "setupversion_lumi", "GT setup version vs LS", TotalNrBinsLs, 0., totalNrBinsLs, 100, -0.1, 1.e10, "i");
  setupversion_lumi->setAxisTitle("Luminosity segment", 1);
  setupversion_lumi->setAxisTitle("Setup version", 2);

  gtfe_bx = ibooker.book1D("gtfe_bx", "GTFE Bx number", 3600, 0., 3600.);
  gtfe_bx->setAxisTitle("GTFE BX number", 1);

  dbx_module =
      ibooker.bookProfile("dbx_module", "delta Bx of GT modules wrt GTFE", 20, 0., 20., 100, -4000., 4000., "i");
  dbx_module->setAxisTitle("GT crate module", 1);
  dbx_module->setAxisTitle("Module Bx - GTFE Bx", 2);
  dbx_module->setBinLabel(1, "GTFEevm", 1);
  dbx_module->setBinLabel(2, "TCS", 1);
  dbx_module->setBinLabel(3, "FDL", 1);
  dbx_module->setBinLabel(4, "FDLloc", 1);
  dbx_module->setBinLabel(5, "PSB9", 1);
  dbx_module->setBinLabel(6, "PSB9loc", 1);
  dbx_module->setBinLabel(7, "PSB13", 1);
  dbx_module->setBinLabel(8, "PSB13loc", 1);
  dbx_module->setBinLabel(9, "PSB14", 1);
  dbx_module->setBinLabel(10, "PSB14loc", 1);
  dbx_module->setBinLabel(11, "PSB15", 1);
  dbx_module->setBinLabel(12, "PSB15loc", 1);
  dbx_module->setBinLabel(13, "PSB19", 1);
  dbx_module->setBinLabel(14, "PSB19loc", 1);
  dbx_module->setBinLabel(15, "PSB20", 1);
  dbx_module->setBinLabel(16, "PSB20loc", 1);
  dbx_module->setBinLabel(17, "PSB21", 1);
  dbx_module->setBinLabel(18, "PSB21loc", 1);
  dbx_module->setBinLabel(19, "GMT", 1);

  BST_MasterStatus =
      ibooker.book2D("BST_MasterStatus", "BST master status over LS", TotalNrBinsLs, 0., totalNrBinsLs, 6, -1., 5.);
  BST_MasterStatus->setAxisTitle("Luminosity segment", 1);
  BST_MasterStatus->setAxisTitle("BST master status", 2);
  BST_MasterStatus->setBinLabel(2, "Master Beam 1", 2);
  BST_MasterStatus->setBinLabel(3, "Master Beam 2", 2);

  BST_turnCountNumber =
      ibooker.book2D("BST_turnCountNumber", "BST turn count over LS", TotalNrBinsLs, 0., totalNrBinsLs, 250, 0., 4.3e9);
  BST_turnCountNumber->setAxisTitle("Luminosity segment", 1);
  BST_turnCountNumber->setAxisTitle("BST turn count number", 2);

  BST_lhcFillNumber = ibooker.book1D("BST_lhcFillNumber", "BST LHC fill number % 1000", 1000, 0., 1000.);
  BST_lhcFillNumber->setAxisTitle("BST LHC fill number modulo 1000");

  BST_beamMode = ibooker.book2D("BST_beamMode", "BST beam mode over LS", TotalNrBinsLs, 0., totalNrBinsLs, 25, 1., 26.);
  BST_beamMode->setAxisTitle("Luminosity segment", 1);
  BST_beamMode->setAxisTitle("Mode", 2);
  BST_beamMode->setBinLabel(1, "No mode", 2);
  BST_beamMode->setBinLabel(2, "Setup", 2);
  BST_beamMode->setBinLabel(3, "Inj pilot", 2);
  BST_beamMode->setBinLabel(4, "Inj intr", 2);
  BST_beamMode->setBinLabel(5, "Inj nomn", 2);
  BST_beamMode->setBinLabel(6, "Pre ramp", 2);
  BST_beamMode->setBinLabel(7, "Ramp", 2);
  BST_beamMode->setBinLabel(8, "Flat top", 2);
  BST_beamMode->setBinLabel(9, "Squeeze", 2);
  BST_beamMode->setBinLabel(10, "Adjust", 2);
  BST_beamMode->setBinLabel(11, "Stable", 2);
  BST_beamMode->setBinLabel(12, "Unstable", 2);
  BST_beamMode->setBinLabel(13, "Beam dump", 2);
  BST_beamMode->setBinLabel(14, "Ramp down", 2);
  BST_beamMode->setBinLabel(15, "Recovery", 2);
  BST_beamMode->setBinLabel(16, "Inj dump", 2);
  BST_beamMode->setBinLabel(17, "Circ dump", 2);
  BST_beamMode->setBinLabel(18, "Abort", 2);
  BST_beamMode->setBinLabel(19, "Cycling", 2);
  BST_beamMode->setBinLabel(20, "Warn beam dump", 2);
  BST_beamMode->setBinLabel(21, "No beam", 2);

  BST_beamMomentum =
      ibooker.book2D("BST_beamMomentum", "BST beam momentum", TotalNrBinsLs, 0., totalNrBinsLs, 100, 0., 7200.);
  BST_beamMomentum->setAxisTitle("Luminosity segment", 1);
  BST_beamMomentum->setAxisTitle("Beam momentum", 2);

  gpsfreq = ibooker.book1D("gpsfreq", "Clock frequency measured by GPS", 1000, 39.95, 40.2);
  gpsfreq->setAxisTitle("CMS clock frequency (MHz)");

  gpsfreqwide = ibooker.book1D("gpsfreqwide", "Clock frequency measured by GPS", 1000, -2., 200.);
  gpsfreqwide->setAxisTitle("CMS clock frequency (MHz)");

  gpsfreqlum = ibooker.book2D(
      "gpsfreqlum", "Clock frequency measured by GPS", TotalNrBinsLs, 0., totalNrBinsLs, 100, 39.95, 40.2);
  gpsfreqlum->setAxisTitle("Luminosity segment", 1);
  gpsfreqlum->setAxisTitle("CMS clock frequency (MHz)", 2);

  BST_intensityBeam1 =
      ibooker.book2D("BST_intensityBeam1", "Intensity beam 1", TotalNrBinsLs, 0., totalNrBinsLs, 1000, 0., 5000.);
  BST_intensityBeam1->setAxisTitle("Luminosity segment", 1);
  BST_intensityBeam1->setAxisTitle("Beam intensity", 2);

  BST_intensityBeam2 =
      ibooker.book2D("BST_intensityBeam2", "Intensity beam 2", TotalNrBinsLs, 0., totalNrBinsLs, 1000, 0., 5000.);
  BST_intensityBeam2->setAxisTitle("Luminosity segment", 1);
  BST_intensityBeam2->setAxisTitle("Beam intensity", 2);

  // prescale factor index monitoring

  m_monL1PrescaleFactorSet = ibooker.book2D(
      "L1PrescaleFactorSet", "Index of L1 prescale factor set", TotalNrBinsLs, 0., totalNrBinsLs, 25, 0., 25.);
  m_monL1PrescaleFactorSet->setAxisTitle("Luminosity segment", 1);
  m_monL1PrescaleFactorSet->setAxisTitle("L1 PF set index", 2);
  m_monL1PfIndicesPerLs =
      ibooker.book1D("L1PfIndicesPerLs", "Number of prescale factor indices used per LS", 10, 0., 10.);
  m_monL1PfIndicesPerLs->setAxisTitle("Number of PF indices used per LS", 1);
  m_monL1PfIndicesPerLs->setAxisTitle("Entries", 2);

  // TCS vs FDL common quantity monitoring

  ibooker.setCurrentFolder(m_histFolder + "/TCSvsEvmFDL");

  //    orbit number
  m_monOrbitNrDiffTcsFdlEvm = ibooker.book1D("OrbitNrDiffTcsFdlEvm",
                                             "Orbit number difference (TCS - EVM_FDL)",
                                             2 * MaxOrbitNrDiffTcsFdlEvm + 1,
                                             static_cast<float>(-(MaxOrbitNrDiffTcsFdlEvm + 1)),
                                             static_cast<float>(MaxOrbitNrDiffTcsFdlEvm + 1));
  m_monOrbitNrDiffTcsFdlEvm->setAxisTitle("Orbit number difference", 1);
  m_monOrbitNrDiffTcsFdlEvm->setAxisTitle("Entries/run", 2);
  m_monLsNrDiffTcsFdlEvm = ibooker.book1D("LsNrDiffTcsFdlEvm",
                                          "LS number difference (TCS - EVM_FDL)",
                                          2 * MaxLsNrDiffTcsFdlEvm + 1,
                                          static_cast<float>(-(MaxLsNrDiffTcsFdlEvm + 1)),
                                          static_cast<float>(MaxLsNrDiffTcsFdlEvm + 1));
  m_monLsNrDiffTcsFdlEvm->setAxisTitle("LS number difference", 1);
  m_monLsNrDiffTcsFdlEvm->setAxisTitle("Entries/run", 2);

  //    LS number
  m_monOrbitNrDiffTcsFdlEvmLs = ibooker.book2D("OrbitNrDiffTcsFdlEvmLs",
                                               "Orbit number difference (TCS - EVM_FDL)",
                                               TotalNrBinsLs,
                                               0.,
                                               totalNrBinsLs,
                                               2 * MaxOrbitNrDiffTcsFdlEvm + 1,
                                               static_cast<float>(-(MaxOrbitNrDiffTcsFdlEvm + 1)),
                                               static_cast<float>(MaxOrbitNrDiffTcsFdlEvm + 1));
  m_monOrbitNrDiffTcsFdlEvmLs->setAxisTitle("Luminosity segment", 1);
  m_monOrbitNrDiffTcsFdlEvmLs->setAxisTitle("Orbit number difference (TCS - EVM_FDL)", 2);

  m_monLsNrDiffTcsFdlEvmLs = ibooker.book2D("LsNrDiffTcsFdlEvmLs",
                                            "LS number difference (TCS - EVM_FDL)",
                                            TotalNrBinsLs,
                                            0.,
                                            totalNrBinsLs,
                                            2 * MaxLsNrDiffTcsFdlEvm + 1,
                                            static_cast<float>(-(MaxLsNrDiffTcsFdlEvm + 1)),
                                            static_cast<float>(MaxLsNrDiffTcsFdlEvm + 1));
  m_monLsNrDiffTcsFdlEvmLs->setAxisTitle("Luminosity segment", 1);
  m_monLsNrDiffTcsFdlEvmLs->setAxisTitle("LS number difference (TCS - EVM_FDL)", 2);

  ibooker.setCurrentFolder(m_histFolder);
  // clear bookkeeping for prescale factor change
  m_pairLsNumberPfIndex.clear();

  ibooker.setCurrentFolder(m_histFolder + "/PlotTrigsBx");

  //--------book AlgoBits/TechBits vs Bx Histogram-----------

  //edm::ESHandle<L1GtTriggerMenu> menuRcd;
  //evSetup.get<L1GtTriggerMenuRcd>().get(menuRcd);
  //menuRcd.product();
  const L1GtTriggerMenu* menu = &evSetup.getData(l1gtTrigmenuToken_);

  h_L1AlgoBX1 = ibooker.book2D("h_L1AlgoBX1", "L1 Algo Trigger BX (algo bit 0 to 31)", 32, -0.5, 31.5, 5, -2.5, 2.5);
  h_L1AlgoBX2 = ibooker.book2D("h_L1AlgoBX2", "L1 Algo Trigger BX (algo bit 32 to 63)", 32, 31.5, 63.5, 5, -2.5, 2.5);
  h_L1AlgoBX3 = ibooker.book2D("h_L1AlgoBX3", "L1 Algo Trigger BX (algo bit 64 to 95)", 32, 63.5, 95.5, 5, -2.5, 2.5);
  h_L1AlgoBX4 = ibooker.book2D("h_L1AlgoBX4", "L1 Algo Trigger BX (algo bit 96 to 127)", 32, 95.5, 127.5, 5, -2.5, 2.5);
  h_L1TechBX = ibooker.book2D("h_L1TechBX", "L1 Tech Trigger BX", 64, -0.5, 63.5, 5, -2.5, 2.5);

  for (CItAlgo algo = menu->gtAlgorithmMap().begin(); algo != menu->gtAlgorithmMap().end(); ++algo) {
    int itrig = (algo->second).algoBitNumber();
    //algoBitToName[itrig] = TString( (algo->second).algoName() );
    //const char* trigName =  (algo->second).algoName().c_str();
    if (itrig < 32) {
      //h_L1AlgoBX1->setBinLabel(itrig+1,trigName);
      h_L1AlgoBX1->setBinLabel(itrig + 1, std::to_string(itrig));
      h_L1AlgoBX1->setAxisTitle("Algorithm trigger bits", 1);
      h_L1AlgoBX1->setAxisTitle("BX (0=L1A)", 2);
    } else if (itrig < 64) {
      //h_L1AlgoBX2->setBinLabel(itrig+1-32,trigName);
      h_L1AlgoBX2->setBinLabel(itrig + 1 - 32, std::to_string(itrig));
      h_L1AlgoBX2->setAxisTitle("Algorithm trigger bits", 1);
      h_L1AlgoBX2->setAxisTitle("BX (0=L1A)", 2);
    } else if (itrig < 96) {
      //h_L1AlgoBX3->setBinLabel(itrig+1-64,trigName);
      h_L1AlgoBX3->setBinLabel(itrig + 1 - 64, std::to_string(itrig));
      h_L1AlgoBX3->setAxisTitle("Algorithm trigger bits", 1);
      h_L1AlgoBX3->setAxisTitle("BX (0=L1A)", 2);
    } else if (itrig < 128) {
      //h_L1AlgoBX4->setBinLabel(itrig+1-96,trigName);
      h_L1AlgoBX4->setBinLabel(itrig + 1 - 96, std::to_string(itrig));
      h_L1AlgoBX4->setAxisTitle("Algorithm trigger bits", 1);
      h_L1AlgoBX4->setAxisTitle("BX (0=L1A)", 2);
    }
  }

  // technical trigger bits
  for (CItAlgo techTrig = menu->gtTechnicalTriggerMap().begin(); techTrig != menu->gtTechnicalTriggerMap().end();
       ++techTrig) {
    int itrig = (techTrig->second).algoBitNumber();
    //techBitToName[itrig] = TString( (techTrig->second).algoName() );
    //const char* trigName =  (techTrig->second).algoName().c_str();
    h_L1TechBX->setBinLabel(itrig + 1, std::to_string(itrig));
    h_L1TechBX->setAxisTitle("Technical trigger bits", 1);
    h_L1TechBX->setAxisTitle("BX (0=L1A)", 2);
  }
}

void L1TGT::dqmBeginRun(edm::Run const& iRrun, edm::EventSetup const& evSetup) { m_nrEvRun = 0; }

//
void L1TGT::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  m_nrEvJob++;

  if (verbose_) {
    edm::LogInfo("L1TGT") << "L1TGT: analyze...." << std::endl;
  }

  // initialize Bx, orbit number, luminosity segment number to invalid value
  int tcsBx = -1;
  int gtfeEvmBx = -1;

  long long int orbitTcs = -1;
  int orbitEvmFdl = -1;

  int lsTcs = -1;
  int lsEvmFdl = -1;

  // get once only the LS block number, to be used in many histograms
  const int lsNumber = iEvent.luminosityBlock();

  // open EVM readout record if available
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
  iEvent.getByToken(gtEvmSource_, gtEvmReadoutRecord);

  if (!gtEvmReadoutRecord.isValid()) {
    edm::LogInfo("L1TGT") << "can't find L1GlobalTriggerEvmReadoutRecord";
  } else {
    // get all info from the EVM record if available and fill the histograms

    const L1GtfeWord& gtfeEvmWord = gtEvmReadoutRecord->gtfeWord();
    const L1GtfeExtWord& gtfeEvmExtWord = gtEvmReadoutRecord->gtfeWord();

    gtfeEvmBx = gtfeEvmWord.bxNr();
    int gtfeEvmActiveBoards = gtfeEvmWord.activeBoards();

    if (isActive(gtfeEvmActiveBoards, TCS)) {
      // if TCS present in the record

      const L1TcsWord& tcsWord = gtEvmReadoutRecord->tcsWord();

      tcsBx = tcsWord.bxNr();
      orbitTcs = tcsWord.orbitNr();
      lsTcs = tcsWord.luminositySegmentNr();

      event_type->Fill(tcsWord.triggerType());
      orbit_lumi->Fill(lsNumber, orbitTcs);

      trigger_number->Fill(tcsWord.partTrigNr());
      event_number->Fill(tcsWord.eventNr());

      trigger_lumi->Fill(lsNumber, tcsWord.partTrigNr());
      event_lumi->Fill(lsNumber, tcsWord.eventNr());
      evnum_trignum_lumi->Fill(lsNumber,
                               static_cast<double>(tcsWord.eventNr()) / static_cast<double>(tcsWord.partTrigNr()));

      uint16_t master = gtfeEvmExtWord.bstMasterStatus();
      uint32_t turnCount = gtfeEvmExtWord.turnCountNumber();
      uint32_t lhcFill = gtfeEvmExtWord.lhcFillNumber();
      uint16_t beam = gtfeEvmExtWord.beamMode();
      uint16_t momentum = gtfeEvmExtWord.beamMomentum();
      uint32_t intensity1 = gtfeEvmExtWord.totalIntensityBeam1();
      uint32_t intensity2 = gtfeEvmExtWord.totalIntensityBeam2();

      BST_MasterStatus->Fill(lsNumber, static_cast<double>(master));
      BST_turnCountNumber->Fill(lsNumber, static_cast<double>(turnCount));
      BST_lhcFillNumber->Fill(static_cast<double>(lhcFill % 1000));
      BST_beamMode->Fill(lsNumber, static_cast<double>(beam));

      BST_beamMomentum->Fill(lsNumber, static_cast<double>(momentum));
      BST_intensityBeam1->Fill(lsNumber, static_cast<double>(intensity1));
      BST_intensityBeam2->Fill(lsNumber, static_cast<double>(intensity2));

      if (verbose_) {
        edm::LogInfo("L1TGT") << " check mode = " << beam << "    momentum " << momentum << " int2 " << intensity2
                              << std::endl;
      }

      uint64_t gpsr = gtfeEvmExtWord.gpsTime();
      uint64_t gpshi = (gpsr >> 32) & 0xffffffff;
      uint64_t gpslo = gpsr & 0xffffffff;
      uint64_t gps = gpshi * 1000000 + gpslo;
      //  edm::LogInfo("L1TGT") << "  gpsr = " << std::hex << gpsr << " hi=" << gpshi << " lo=" << gpslo << " gps=" << gps << std::endl;

      Long64_t delorb = orbitTcs - preOrb_;
      Long64_t delgps = gps - preGps_;
      Double_t freq = -1.;

      if (delgps > 0) {
        freq = ((Double_t)(delorb)) * 3564. / ((Double_t)(delgps));
      }

      if (delorb > 0) {
        gpsfreq->Fill(freq);
        gpsfreqwide->Fill(freq);
        gpsfreqlum->Fill(lsNumber, freq);
        if (verbose_) {
          if (freq > 200.) {
            edm::LogInfo("L1TGT") << " preOrb_ = " << preOrb_ << " orbitTcs=" << orbitTcs << " delorb=" << delorb
                                  << std::hex << " preGps_=" << preGps_ << " gps=" << gps << std::dec
                                  << " delgps=" << delgps << " freq=" << freq << std::endl;
          }
        }
      }

      preGps_ = gps;
      preOrb_ = orbitTcs;
    }

    // get info from FDL if active
    if (isActive(gtfeEvmActiveBoards, FDL)) {
      const L1GtFdlWord& fdlWord = gtEvmReadoutRecord->gtFdlWord();

      orbitEvmFdl = fdlWord.orbitNr();
      lsEvmFdl = fdlWord.lumiSegmentNr();
    }

    if ((orbitTcs >= 0) && (orbitEvmFdl >= 0)) {
      int diffOrbit = static_cast<float>(orbitTcs - orbitEvmFdl);
      edm::LogInfo("L1TGT") << "\n orbitTcs = " << orbitTcs << " orbitEvmFdl = " << orbitEvmFdl
                            << " diffOrbit = " << diffOrbit << " orbitEvent = " << iEvent.orbitNumber() << std::endl;

      if (diffOrbit >= MaxOrbitNrDiffTcsFdlEvm) {
        m_monOrbitNrDiffTcsFdlEvm->Fill(MaxOrbitNrDiffTcsFdlEvm);

      } else if (diffOrbit <= -MaxOrbitNrDiffTcsFdlEvm) {
        m_monOrbitNrDiffTcsFdlEvm->Fill(-MaxOrbitNrDiffTcsFdlEvm);

      } else {
        m_monOrbitNrDiffTcsFdlEvm->Fill(diffOrbit);
        m_monOrbitNrDiffTcsFdlEvmLs->Fill(lsNumber, diffOrbit);
      }

    } else {
      if (orbitTcs >= 0) {
        // EVM_FDL error
        m_monOrbitNrDiffTcsFdlEvm->Fill(MaxOrbitNrDiffTcsFdlEvm);
      } else if (orbitEvmFdl >= 0) {
        // TCS error
        m_monOrbitNrDiffTcsFdlEvm->Fill(-MaxOrbitNrDiffTcsFdlEvm);

      } else {
        // TCS and EVM_FDL error
        m_monOrbitNrDiffTcsFdlEvm->Fill(-MaxOrbitNrDiffTcsFdlEvm);
        m_monOrbitNrDiffTcsFdlEvm->Fill(MaxOrbitNrDiffTcsFdlEvm);
      }
    }

    if ((lsTcs >= 0) && (lsEvmFdl >= 0)) {
      int diffLs = static_cast<float>(lsTcs - lsEvmFdl);
      edm::LogInfo("L1TGT") << "\n lsTcs = " << lsTcs << " lsEvmFdl = " << lsEvmFdl << " diffLs = " << diffLs
                            << " lsEvent = " << lsNumber << std::endl;

      if (diffLs >= MaxLsNrDiffTcsFdlEvm) {
        m_monLsNrDiffTcsFdlEvm->Fill(MaxLsNrDiffTcsFdlEvm);

      } else if (diffLs <= -MaxLsNrDiffTcsFdlEvm) {
        m_monLsNrDiffTcsFdlEvm->Fill(-MaxLsNrDiffTcsFdlEvm);

      } else {
        m_monLsNrDiffTcsFdlEvm->Fill(diffLs);
        m_monLsNrDiffTcsFdlEvmLs->Fill(lsNumber, diffLs);
      }

    } else {
      if (lsTcs >= 0) {
        // EVM_FDL error
        m_monLsNrDiffTcsFdlEvm->Fill(MaxLsNrDiffTcsFdlEvm);
      } else if (lsEvmFdl >= 0) {
        // TCS error
        m_monLsNrDiffTcsFdlEvm->Fill(-MaxLsNrDiffTcsFdlEvm);

      } else {
        // TCS and EVM_FDL error
        m_monLsNrDiffTcsFdlEvm->Fill(-MaxLsNrDiffTcsFdlEvm);
        m_monLsNrDiffTcsFdlEvm->Fill(MaxLsNrDiffTcsFdlEvm);
      }
    }
  }

  // open GT DAQ readout record - exit if failed
  edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
  iEvent.getByToken(gtSource_L1GT_, gtReadoutRecord);

  //edm::ESHandle<L1GtTriggerMenu> menuRcd;
  //evSetup.get<L1GtTriggerMenuRcd>().get(menuRcd);

  //const L1GtTriggerMenu* menu = menuRcd.product();

  if (!gtReadoutRecord.isValid()) {
    edm::LogInfo("L1TGT") << "can't find L1GlobalTriggerReadoutRecord";
    return;
  }

  if (gtReadoutRecord.isValid()) {
    unsigned int NmaxL1AlgoBit = gtReadoutRecord->decisionWord().size();
    unsigned int NmaxL1TechBit = gtReadoutRecord->technicalTriggerWord().size();

    const DecisionWord dWord = gtReadoutRecord->decisionWord();
    const TechnicalTriggerWord technicalTriggerWordBeforeMask = gtReadoutRecord->technicalTriggerWord();

    const std::vector<L1GtFdlWord>& m_gtFdlWord(gtReadoutRecord->gtFdlVector());
    int numberBxInEvent = m_gtFdlWord.size();
    int minBxInEvent = (numberBxInEvent + 1) / 2 - numberBxInEvent;

    for (unsigned int iBit = 0; iBit < NmaxL1AlgoBit; ++iBit) {
      bool accept = dWord[iBit];

      typedef std::map<std::string, bool>::value_type valType;
      trig_iter = l1TriggerDecision.find(algoBitToName[iBit]);
      if (trig_iter == l1TriggerDecision.end()) {
        l1TriggerDecision.insert(valType(algoBitToName[iBit], accept));
      } else {
        trig_iter->second = accept;
      }

      int ibx = 0;
      for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); itBx != m_gtFdlWord.end(); ++itBx) {
        const DecisionWord dWordBX = (*itBx).gtDecisionWord();
        bool accept = dWordBX[iBit];
        if (accept) {
          if (iBit < 32)
            h_L1AlgoBX1->Fill(iBit, minBxInEvent + ibx);
          else if (iBit < 64)
            h_L1AlgoBX2->Fill(iBit, minBxInEvent + ibx);
          else if (iBit < 96)
            h_L1AlgoBX3->Fill(iBit, minBxInEvent + ibx);
          else if (iBit < 128)
            h_L1AlgoBX4->Fill(iBit, minBxInEvent + ibx);
        }
        ibx++;
      }
    }

    for (unsigned int iBit = 0; iBit < NmaxL1TechBit; ++iBit) {
      bool accept = technicalTriggerWordBeforeMask[iBit];

      typedef std::map<std::string, bool>::value_type valType;
      trig_iter = l1TechTriggerDecision.find(techBitToName[iBit]);
      if (trig_iter == l1TechTriggerDecision.end())
        l1TechTriggerDecision.insert(valType(techBitToName[iBit], accept));
      else
        trig_iter->second = accept;

      int ibx = 0;
      for (std::vector<L1GtFdlWord>::const_iterator itBx = m_gtFdlWord.begin(); itBx != m_gtFdlWord.end(); ++itBx) {
        const DecisionWord dWordBX = (*itBx).gtTechnicalTriggerWord();
        bool accept = dWordBX[iBit];
        if (accept)
          h_L1TechBX->Fill(iBit, minBxInEvent + ibx);
        ibx++;
      }
    }
  }

  // initialize bx's to invalid value
  int gtfeBx = -1;
  int fdlBx[2] = {-1, -1};
  int psbBx[2][7] = {{-1, -1, -1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1, -1, -1}};
  int gmtBx = -1;

  // get info from GTFE DAQ record
  const L1GtfeWord& gtfeWord = gtReadoutRecord->gtfeWord();
  gtfeBx = gtfeWord.bxNr();
  gtfe_bx->Fill(gtfeBx);
  setupversion_lumi->Fill(lsNumber, gtfeWord.setupVersion());
  int gtfeActiveBoards = gtfeWord.activeBoards();

  // look for GMT readout collection from the same source if GMT active
  if (isActive(gtfeActiveBoards, GMT)) {
    edm::Handle<L1MuGMTReadoutCollection> gmtReadoutCollection;
    iEvent.getByToken(gtSource_L1MuGMT_, gmtReadoutCollection);

    if (gmtReadoutCollection.isValid()) {
      gmtBx = gmtReadoutCollection->getRecord().getBxNr();
    }
  }

  // get info from FDL if active (including decision word)
  if (isActive(gtfeActiveBoards, FDL)) {
    const L1GtFdlWord& fdlWord = gtReadoutRecord->gtFdlWord();
    fdlBx[0] = fdlWord.bxNr();
    fdlBx[1] = fdlWord.localBxNr();

    /// get Global Trigger algo and technical trigger bit statistics
    const DecisionWord& gtDecisionWord = gtReadoutRecord->decisionWord();
    const TechnicalTriggerWord& gtTTWord = gtReadoutRecord->technicalTriggerWord();

    int dbitNumber = 0;
    DecisionWord::const_iterator GTdbitItr;
    algo_bits->Fill(-1.);  // fill underflow to normalize
    for (GTdbitItr = gtDecisionWord.begin(); GTdbitItr != gtDecisionWord.end(); GTdbitItr++) {
      if (*GTdbitItr) {
        algo_bits->Fill(dbitNumber);
        algo_bits_lumi->Fill(lsNumber, dbitNumber);
        int dbitNumber1 = 0;
        DecisionWord::const_iterator GTdbitItr1;
        for (GTdbitItr1 = gtDecisionWord.begin(); GTdbitItr1 != gtDecisionWord.end(); GTdbitItr1++) {
          if (*GTdbitItr1)
            algo_bits_corr->Fill(dbitNumber, dbitNumber1);
          dbitNumber1++;
        }
        int tbitNumber1 = 0;
        TechnicalTriggerWord::const_iterator GTtbitItr1;
        for (GTtbitItr1 = gtTTWord.begin(); GTtbitItr1 != gtTTWord.end(); GTtbitItr1++) {
          if (*GTtbitItr1)
            algo_tt_bits_corr->Fill(dbitNumber, tbitNumber1);
          tbitNumber1++;
        }
      }
      dbitNumber++;
    }

    int tbitNumber = 0;
    TechnicalTriggerWord::const_iterator GTtbitItr;
    tt_bits->Fill(-1.);  // fill underflow to normalize
    for (GTtbitItr = gtTTWord.begin(); GTtbitItr != gtTTWord.end(); GTtbitItr++) {
      if (*GTtbitItr) {
        tt_bits->Fill(tbitNumber);
        tt_bits_lumi->Fill(lsNumber, tbitNumber);
        int tbitNumber1 = 0;
        TechnicalTriggerWord::const_iterator GTtbitItr1;
        for (GTtbitItr1 = gtTTWord.begin(); GTtbitItr1 != gtTTWord.end(); GTtbitItr1++) {
          if (*GTtbitItr1)
            tt_bits_corr->Fill(tbitNumber, tbitNumber1);
          tbitNumber1++;
        }
      }
      tbitNumber++;
    }

    // fill the index of actual prescale factor set
    // the index for technical triggers and algorithm trigger is the same (constraint in L1 GT TS)
    // so we read only pfIndexAlgoTrig (uint16_t)

    const int pfIndexAlgoTrig = fdlWord.gtPrescaleFactorIndexAlgo();
    m_monL1PrescaleFactorSet->Fill(lsNumber, static_cast<float>(pfIndexAlgoTrig));

    //

    // check that the combination (lsNumber, pfIndex) is not already included
    // to avoid fake entries due to different event order

    std::pair<int, int> pairLsPfi = std::make_pair(lsNumber, pfIndexAlgoTrig);

    CItVecPair cIt = find(m_pairLsNumberPfIndex.begin(), m_pairLsNumberPfIndex.end(), pairLsPfi);

    if (cIt == m_pairLsNumberPfIndex.end()) {
      m_pairLsNumberPfIndex.push_back(pairLsPfi);
    }
  }

  // get info from active PSB's
  int ibit = PSB9;  // first psb
  // for now hardcode psb id's - TODO - get them from Vasile's board maps...
  int psbID[7] = {0xbb09, 0xbb0d, 0xbb0e, 0xbb0f, 0xbb13, 0xbb14, 0xbb15};
  for (int i = 0; i < 7; i++) {
    if (isActive(gtfeActiveBoards, ibit)) {
      L1GtPsbWord psbWord = gtReadoutRecord->gtPsbWord(psbID[i]);
      psbBx[0][i] = psbWord.bxNr();
      psbBx[1][i] = psbWord.localBxNr();
    }
    ibit++;
  }

  //fill the dbx histo
  if (gtfeEvmBx > -1)
    dbx_module->Fill(0., gtfeEvmBx - gtfeBx);
  if (tcsBx > -1)
    dbx_module->Fill(1., tcsBx - gtfeBx);
  for (int i = 0; i < 2; i++) {
    if (fdlBx[i] > -1)
      dbx_module->Fill(2. + i, fdlBx[i] - gtfeBx);
  }
  for (int j = 0; j < 7; j++) {
    for (int i = 0; i < 2; i++) {
      if (psbBx[i][j] > -1)
        dbx_module->Fill(4. + i + 2 * j, psbBx[i][j] - gtfeBx);
    }
  }
  if (gmtBx > -1)
    dbx_module->Fill(18., gmtBx - gtfeBx);
}

// end section
void L1TGT::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& evSetup) {
  if (m_runInEndLumi) {
    countPfsIndicesPerLs();
  }
}

////////////////////////////////////////////////////////////////////////////////////////
bool L1TGT::isActive(int word, int bit) {
  if (word & (1 << bit))
    return true;
  return false;
}

void L1TGT::countPfsIndicesPerLs() {
  if (verbose_) {
    edm::LogInfo("L1TGT") << "\n  Prescale factor indices used in a LS " << std::endl;

    for (CItVecPair cIt = m_pairLsNumberPfIndex.begin(); cIt != m_pairLsNumberPfIndex.end(); ++cIt) {
      edm::LogVerbatim("L1TGT") << "  lsNumber = " << (*cIt).first << " pfIndex = " << (*cIt).second << std::endl;
    }
    edm::LogVerbatim("L1TGT") << std::endl;
  }

  // reset the histogram...
  m_monL1PfIndicesPerLs->Reset();

  // sort the vector (for pairs: sort after first argument, then after the second argument)
  std::sort(m_pairLsNumberPfIndex.begin(), m_pairLsNumberPfIndex.end());

  int previousLsNumber = -1;
  int previousPfsIndex = -1;

  // count the number of pairs (lsNumber, pfIndex) per Ls
  // there are no duplicate entries, and pairs are sorted after both members
  // ... and fill again the histogram
  for (CItVecPair cIt = m_pairLsNumberPfIndex.begin(); cIt != m_pairLsNumberPfIndex.end(); ++cIt) {
    int pfsIndicesPerLs = 1;

    if ((*cIt).first == previousLsNumber) {
      if ((*cIt).second != previousPfsIndex) {
        pfsIndicesPerLs++;
        previousPfsIndex = (*cIt).second;
      }

    } else {
      // fill the histogram with the number of PF indices for the previous Ls
      if (previousLsNumber != -1) {
        m_monL1PfIndicesPerLs->Fill(pfsIndicesPerLs);
      }

      // new Ls
      previousLsNumber = (*cIt).first;
      previousPfsIndex = (*cIt).second;

      pfsIndicesPerLs = 1;
    }
  }
}

// static class members
// maximum difference in orbit number
const int L1TGT::MaxOrbitNrDiffTcsFdlEvm = 24;

// maximum difference in luminosity segment number
const int L1TGT::MaxLsNrDiffTcsFdlEvm = 24;
