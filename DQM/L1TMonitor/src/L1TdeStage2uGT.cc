/*
 * \file L1TdeStage2uGT.cc
 *
 * L. Apanasevich <Leonard.Apanasevich@cern.ch>
 */

#include "DQM/L1TMonitor/interface/L1TdeStage2uGT.h"

L1TdeStage2uGT::L1TdeStage2uGT(const edm::ParameterSet& ps)
    : dataLabel_(ps.getParameter<edm::InputTag>("dataSource")),
      dataSource_(consumes<GlobalAlgBlkBxCollection>(dataLabel_)),
      emulLabel_(ps.getParameter<edm::InputTag>("emulSource")),
      emulSource_(consumes<GlobalAlgBlkBxCollection>(emulLabel_)),
      triggerBlackList_(ps.getParameter<std::vector<std::string> >("triggerBlackList")),
      numBx_(ps.getParameter<int>("numBxToMonitor")),
      histFolder_(ps.getParameter<std::string>("histFolder")),
      gtUtil_(ps,
              consumesCollector(),
              *this,
              ps.getParameter<edm::InputTag>("dataSource"),
              ps.getParameter<edm::InputTag>("dataSource"),
              l1t::UseEventSetupIn::RunAndEvent),
      numLS_(2000),
      m_currentLumi(0),
      m_currentRun(0) {
  if (numBx_ > 5)
    numBx_ = 5;
  if ((numBx_ > 0) && ((numBx_ % 2) == 0)) {
    numBx_ = numBx_ - 1;

    edm::LogWarning("L1TdeStage2uGT") << "\nWARNING: Number of bunch crossing to be emulated rounded to: " << numBx_
                                      << "\n         The number must be an odd number!\n"
                                      << std::endl;
  }
  firstBx = (numBx_ + 1) / 2 - numBx_;
  lastBx = (numBx_ + 1) / 2 - 1;

  edm::LogInfo("L1TdeStage2uGT") << "Number of bunches crossings monitored: " << numBx_ << "\t"
                                 << "Min BX= " << firstBx << "\t"
                                 << "Max BX= " << lastBx << std::endl;
}

L1TdeStage2uGT::~L1TdeStage2uGT() {}

void L1TdeStage2uGT::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<GlobalAlgBlkBxCollection> dataCollection;
  event.getByToken(dataSource_, dataCollection);
  edm::Handle<GlobalAlgBlkBxCollection> emulCollection;
  event.getByToken(emulSource_, emulCollection);

  if (!dataCollection.isValid()) {
    edm::LogError("L1TdeStage2uGT") << "Cannot find unpacked uGT readout record.";
    return;
  }
  if (!emulCollection.isValid()) {
    edm::LogError("L1TdeStage2uGT") << "Cannot find emulated uGT readout record.";
    return;
  }

  // Only using gtUtil to find prescale factors and mapping of bits to names, so only call gtUtil_ at lumi boundaries.
  if (m_currentLumi != event.luminosityBlock() || m_currentRun != event.run()) {
    m_currentLumi = event.luminosityBlock();
    m_currentRun = event.run();
    gtUtil_.retrieveL1(event, es, dataSource_);
  }

  // Get standard event parameters
  int lumi = event.luminosityBlock();
  if (lumi > numLS_)
    lumi = numLS_;

  // int bx = event.bunchCrossing();

  // check that the requested range of BX's is consistent with the BX's in the emulated and unpacked collections
  if (emulCollection->getFirstBX() > firstBx)
    firstBx = emulCollection->getFirstBX();
  if (emulCollection->getLastBX() < lastBx)
    lastBx = emulCollection->getLastBX();

  if (dataCollection->getFirstBX() > firstBx)
    firstBx = dataCollection->getFirstBX();
  if (dataCollection->getLastBX() < lastBx)
    lastBx = dataCollection->getLastBX();

  m_normalizationHisto->Fill(float(NInitalMismatchDataNoEmul));
  m_normalizationHisto->Fill(float(NInitalMismatchEmulNoData));
  m_normalizationHisto->Fill(float(NFinalMismatchDataNoEmul));
  m_normalizationHisto->Fill(float(NFinalMismatchEmulNoData));

  for (int ibx = firstBx; ibx <= lastBx; ++ibx) {
    ostringstream bxt;
    if (ibx == 0) {
      bxt << "CentralBX";
    } else {
      bxt << "BX" << ibx;
    }
    std::string hname, hsummary;

    hsummary = "dataEmulSummary_" + bxt.str();

    bool foundInitalMismatchDataNoEmul{false};
    bool foundInitalMismatchEmulNoData{false};
    bool foundFinalMismatchDataNoEmul{false};
    bool foundFinalMismatchEmulNoData{false};
    // Looping over the algo blocks from each trigger board
    std::vector<GlobalAlgBlk>::const_iterator it_data, it_emul;
    for (it_data = dataCollection->begin(ibx), it_emul = emulCollection->begin(ibx);
         it_data != dataCollection->end(ibx) && it_emul != emulCollection->end(ibx);
         ++it_data, ++it_emul) {
      // Fills algorithm bits histograms
      int numAlgs = it_data->getAlgoDecisionInitial().size();
      for (int algoBit = 0; algoBit < numAlgs; ++algoBit) {
        string algoName = "xxx";
        bool found = gtUtil_.getAlgNameFromBit(algoBit, algoName);
        if (not found)
          continue;

        // skip bits which emulator does not handle (only skiped for bx !=0)
        bool isBlackListed(false);
        for (auto const& pattern : triggerBlackList_) {
          //std::cout << pattern << std::endl;
          if (edm::is_glob(pattern)) {
            std::regex regexp(edm::glob2reg(pattern));
            if (regex_match(algoName.c_str(), regexp))
              isBlackListed = true;
          } else {
            if (algoName == pattern)
              isBlackListed = true;
          }
        }
        if (ibx != 0 && isBlackListed)
          continue;

        // Check initial decisions
        if (it_data->getAlgoDecisionInitial(algoBit) != it_emul->getAlgoDecisionInitial(algoBit)) {
          if (it_data->getAlgoDecisionInitial(algoBit)) {
            hname = "DataNoEmul_" + bxt.str();
            foundInitalMismatchDataNoEmul = true;
            initDecisionMismatchesDnoE_vs_LS->Fill(float(lumi));
          } else {
            hname = "EmulatorNoData_" + bxt.str();
            foundInitalMismatchEmulNoData = true;
            initDecisionMismatchesEnoD_vs_LS->Fill(float(lumi));
          }
          fillHist(m_HistNamesInitial, hname, float(algoBit), 1.);
        }

        // Check final decisions
        if (it_data->getAlgoDecisionFinal(algoBit) != it_emul->getAlgoDecisionFinal(algoBit)) {
          bool unprescaled = true;
          // check the prescale factor
          double prescale = -999;
          bool dummy = gtUtil_.getPrescaleByBit(algoBit, prescale);
          if (not dummy)
            edm::LogWarning("L1TdeStage2uGT") << "Could not find prescale value for algobit: " << algoBit << std::endl;

          if (prescale != 1)
            unprescaled = false;

          if (unprescaled) {
            if (it_data->getAlgoDecisionFinal(algoBit)) {
              hname = "DataNoEmul_" + bxt.str();
              foundFinalMismatchDataNoEmul = true;
              finalDecisionMismatchesDnoE_vs_LS->Fill(float(lumi));
            } else {
              hname = "EmulatorNoData_" + bxt.str();
              foundFinalMismatchEmulNoData = true;
              finalDecisionMismatchesEnoD_vs_LS->Fill(float(lumi));
            }
            fillHist(m_HistNamesFinal, hname, float(algoBit), 1.);
          }
        }

      }  // end loop over algoBits
    }    // end loop over globalalgblk vector
    if (foundInitalMismatchDataNoEmul) {
      fillHist(m_SummaryHistograms, hsummary, float(NInitalMismatchDataNoEmul), 1.);
    }
    if (foundInitalMismatchEmulNoData) {
      fillHist(m_SummaryHistograms, hsummary, float(NInitalMismatchEmulNoData), 1.);
    }
    if (foundFinalMismatchDataNoEmul) {
      fillHist(m_SummaryHistograms, hsummary, float(NFinalMismatchDataNoEmul), 1.);
    }
    if (foundFinalMismatchEmulNoData) {
      fillHist(m_SummaryHistograms, hsummary, float(NFinalMismatchEmulNoData), 1.);
    }
  }  // endof loop over BX collections
}

void L1TdeStage2uGT::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup& es) {
  gtUtil_.retrieveL1Setup(es);

  auto const& prescales = gtUtil_.prescales();
  int nbins = prescales.size();  // dummy values for now; update later when gtutils function is called
  double xmin = -0.5;
  double xmax = nbins - 0.5;

  string hname, htitle;

  int ibx = (numBx_ + 1) / 2 - numBx_;
  for (int i = 0; i < numBx_; i++) {
    ostringstream bxn, bxt;

    if (ibx == 0) {
      bxt << "CentralBX";
      bxn << " Central BX ";
    } else {
      bxt << "BX" << ibx;
      bxn << " BX " << ibx;
    }
    ibx++;

    ibooker.setCurrentFolder(histFolder_);
    hname = "dataEmulSummary_" + bxt.str();
    htitle = "uGT Data/Emulator Mismatches --" + bxn.str();
    m_SummaryHistograms[hname] = ibooker.book1D(hname, htitle, NSummaryColumns, 0., double(NSummaryColumns));
    m_SummaryHistograms[hname]->setAxisTitle("Events", /* axis */ 2);
    m_SummaryHistograms[hname]->setBinLabel(1 + NInitalMismatchDataNoEmul, "Data, NoEmul -- Initial Decisions");
    m_SummaryHistograms[hname]->setBinLabel(1 + NInitalMismatchEmulNoData, "Emulator, No Data -- Initial Decisions");
    m_SummaryHistograms[hname]->setBinLabel(1 + NFinalMismatchDataNoEmul, "Data, NoEmul -- Final Decisions");
    m_SummaryHistograms[hname]->setBinLabel(1 + NFinalMismatchEmulNoData, "Emulator, No Data -- Final Decisions");

    if (i == 0) {
      hname = "normalizationHisto";
      htitle = "Normalization histogram for uGT Data/Emulator Mismatches ratios";
      m_normalizationHisto = ibooker.book1D(hname, htitle, NSummaryColumns, 0., double(NSummaryColumns));
      m_normalizationHisto->setAxisTitle("Events", /* axis */ 2);
      m_normalizationHisto->setBinLabel(1 + NInitalMismatchDataNoEmul, "Data, NoEmul -- Initial Decisions");
      m_normalizationHisto->setBinLabel(1 + NInitalMismatchEmulNoData, "Emulator, No Data -- Initial Decisions");
      m_normalizationHisto->setBinLabel(1 + NFinalMismatchDataNoEmul, "Data, NoEmul -- Final Decisions");
      m_normalizationHisto->setBinLabel(1 + NFinalMismatchEmulNoData, "Emulator, No Data -- Final Decisions");
    }

    // book initial decisions histograms
    ibooker.setCurrentFolder(histFolder_ + "/InitialDecisionMismatches");
    initDecisionMismatchesDnoE_vs_LS =
        ibooker.book1D("initialDecisionMismatchesDnoE_vs_LS",
                       "uGT initial decision mismatches (Data, but no emu) vs Luminosity Segment",
                       numLS_,
                       0.,
                       double(numLS_));
    initDecisionMismatchesDnoE_vs_LS->setAxisTitle("Events with Initial Decision Mismatch", /* axis */ 2);
    initDecisionMismatchesDnoE_vs_LS->setAxisTitle("Luminosity Segment");

    initDecisionMismatchesEnoD_vs_LS =
        ibooker.book1D("initialDecisionMismatchesEnoD_vs_LS",
                       "uGT initial decision mismatches (Emu, but no data) vs Luminosity Segment",
                       numLS_,
                       0.,
                       double(numLS_));
    initDecisionMismatchesEnoD_vs_LS->setAxisTitle("Events with Initial Decision Mismatch", /* axis */ 2);
    initDecisionMismatchesEnoD_vs_LS->setAxisTitle("Luminosity Segment");

    hname = "DataNoEmul_" + bxt.str();
    htitle = "uGT data-emul mismatch -- Data fired but not Emulator --" + bxn.str();
    m_HistNamesInitial[hname] = ibooker.book1D(hname, htitle, nbins, xmin, xmax);

    hname = "EmulatorNoData_" + bxt.str();
    htitle = "uGT data-emul mismatch -- Emulator fired but not Data --" + bxn.str();
    m_HistNamesInitial[hname] = ibooker.book1D(hname, htitle, nbins, xmin, xmax);

    // book final decisions histograms
    ibooker.setCurrentFolder(histFolder_ + "/FinalDecisionMismatches");
    finalDecisionMismatchesDnoE_vs_LS =
        ibooker.book1D("finalDecisionMismatchesDnoE_vs_LS",
                       "uGT final decision mismatches (Data, but no emu) vs Luminosity Segment",
                       numLS_,
                       0.,
                       double(numLS_));
    finalDecisionMismatchesDnoE_vs_LS->setAxisTitle("Events with Final Decision Mismatch", /* axis */ 2);
    finalDecisionMismatchesDnoE_vs_LS->setAxisTitle("Luminosity Segment");

    finalDecisionMismatchesEnoD_vs_LS =
        ibooker.book1D("finalDecisionMismatchesEnoD_vs_LS",
                       "uGT final decision mismatches (Emu, but no data) vs Luminosity Segment",
                       numLS_,
                       0.,
                       double(numLS_));
    finalDecisionMismatchesEnoD_vs_LS->setAxisTitle("Events with Final Decision Mismatch", /* axis */ 2);
    finalDecisionMismatchesEnoD_vs_LS->setAxisTitle("Luminosity Segment");

    hname = "DataNoEmul_" + bxt.str();
    htitle = "uGT data-emul mismatch -- Data fired but not Emulator --" + bxn.str();
    m_HistNamesFinal[hname] = ibooker.book1D(hname, htitle, nbins, xmin, xmax);

    hname = "EmulatorNoData_" + bxt.str();
    htitle = "uGT data-emul mismatch -- Emulator fired but not Data --" + bxn.str();
    m_HistNamesFinal[hname] = ibooker.book1D(hname, htitle, nbins, xmin, xmax);
  }

  // Set some histogram attributes
  for (std::map<std::string, MonitorElement*>::iterator it = m_HistNamesInitial.begin(); it != m_HistNamesInitial.end();
       ++it) {
    // for (unsigned int i = 0; i < prescales.size(); i++) {
    //   auto const& name = prescales.at(i).first;
    //   if (name != "NULL")
    // 	(*it).second->setBinLabel(1+i, name.c_str());
    // }
    (*it).second->setAxisTitle("Trigger Bit");
    (*it).second->setAxisTitle("Events with Initial Decision Mismatch", /* axis */ 2);
  }

  for (std::map<std::string, MonitorElement*>::iterator it = m_HistNamesFinal.begin(); it != m_HistNamesFinal.end();
       ++it) {
    // for (unsigned int i = 0; i < prescales.size(); i++) {
    //   auto const& name = prescales.at(i).first;
    //   if (name != "NULL")
    // 	(*it).second->setBinLabel(1+i, name.c_str());
    // }
    (*it).second->setAxisTitle("Trigger Bit (Unprescaled)");
    (*it).second->setAxisTitle("Events with Final Decision Mismatch", /* axis */ 2);
  }
}

void L1TdeStage2uGT::fillHist(const std::map<std::string, MonitorElement*>& m_HistNames,
                              const std::string& histName,
                              const Double_t& value,
                              const Double_t& wt = 1.) {
  std::map<std::string, MonitorElement*>::const_iterator hid = m_HistNames.find(histName);

  if (hid == m_HistNames.end())
    edm::LogWarning("L1TdeStage2uGT") << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(value, wt);
}
