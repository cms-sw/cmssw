/**
 * \class L1TEMTFEventInfoClient
 *
 *
 * Description: see header file.
 *
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 *
 */

// this class header
#include "DQM/L1TMonitorClient/interface/L1TEMTFEventInfoClient.h"

// constructor
L1TEMTFEventInfoClient::L1TEMTFEventInfoClient(const edm::ParameterSet& parSet)
    : m_verbose(parSet.getUntrackedParameter<bool>("verbose", false)),
      m_monitorDir(parSet.getUntrackedParameter<std::string>("monitorDir", "")),
      m_histDir(parSet.getUntrackedParameter<std::string>("histDir", "")),
      m_runInEventLoop(parSet.getUntrackedParameter<bool>("runInEventLoop", false)),
      m_runInEndLumi(parSet.getUntrackedParameter<bool>("runInEndLumi", false)),
      m_runInEndRun(parSet.getUntrackedParameter<bool>("runInEndRun", false)),
      m_runInEndJob(parSet.getUntrackedParameter<bool>("runInEndJob", false)),
      m_trackObjects(parSet.getParameter<std::vector<edm::ParameterSet> >("TrackObjects")),
      m_hitObjects(parSet.getParameter<std::vector<edm::ParameterSet> >("HitObjects")),
      m_disableTrackObjects(parSet.getParameter<std::vector<std::string> >("DisableTrackObjects")),
      m_disableHitObjects(parSet.getParameter<std::vector<std::string> >("DisableHitObjects")),
      m_noisyStrip(parSet.getParameter<std::vector<edm::ParameterSet> >("NoisyStrip")),
      m_deadStrip(parSet.getParameter<std::vector<edm::ParameterSet> >("DeadStrip")),
      m_disableNoisyStrip(parSet.getParameter<std::vector<std::string> >("DisableNoisyStrip")),
      m_disableDeadStrip(parSet.getParameter<std::vector<std::string> >("DisableDeadStrip")),
      m_nrTrackObjects(0),
      m_nrHitObjects(0),
      m_nrNoisyStrip(0),
      m_nrDeadStrip(0),
      m_totalNrQtSummaryEnabled(0) {
  initialize();
}

// destructor
L1TEMTFEventInfoClient::~L1TEMTFEventInfoClient() {
  //empty
}

void L1TEMTFEventInfoClient::initialize() {
  if (m_verbose)
    std::cout << "\nMonitor directory =             " << m_monitorDir << std::endl;

  // L1 systems

  m_nrTrackObjects = m_trackObjects.size();

  m_trackLabel.reserve(m_nrTrackObjects);
  // m_trackLabelExt.reserve(m_nrTrackObjects);  // Not needed? - AWB 05.12.16
  m_trackDisable.reserve(m_nrTrackObjects);

  // on average five quality test per system - just a best guess
  m_trackQualityTestName.reserve(5 * m_nrTrackObjects);   // Not needed? - AWB 05.12.16
  m_trackQualityTestHist.reserve(5 * m_nrTrackObjects);   // Not needed? - AWB 05.12.16
  m_trackQtSummaryEnabled.reserve(5 * m_nrTrackObjects);  // Not needed? - AWB 05.12.16

  int indexSys = 0;

  int totalNrQualityTests = 0;

  for (const auto& itTrack : m_trackObjects) {
    m_trackLabel.push_back(itTrack.getParameter<std::string>("SystemLabel"));

    // m_trackLabelExt.push_back(itTrack.getParameter<std::string>(  // Not needed? - AWB 05.12.16
    //         "HwValLabel"));

    m_trackDisable.push_back(itTrack.getParameter<unsigned int>("SystemDisable"));
    // check the additional disable flag from m_disableTrackObjects
    for (const auto& itSys : m_disableTrackObjects) {
      if (itSys == m_trackLabel[indexSys]) {
        m_trackDisable[indexSys] = 1;
      }
    }

    std::vector<edm::ParameterSet> qTests = itTrack.getParameter<std::vector<edm::ParameterSet> >("QualityTests");
    size_t qtPerSystem = qTests.size();

    std::vector<std::string> qtNames;
    qtNames.reserve(qtPerSystem);

    std::vector<std::string> qtFullPathHists;
    qtFullPathHists.reserve(qtPerSystem);

    std::vector<unsigned int> qtSumEnabled;
    qtSumEnabled.reserve(qtPerSystem);

    if (m_verbose)
      std::cout << "\nLooping over track quality tests" << std::endl;
    for (const auto& itQT : qTests) {
      totalNrQualityTests++;

      qtNames.push_back(itQT.getParameter<std::string>("QualityTestName"));

      // qtFullPathHists.push_back( m_histDir + "/" + itQT.getParameter<std::string> ("QualityTestHist"));
      qtFullPathHists.push_back(itQT.getParameter<std::string>("QualityTestHist"));
      if (m_verbose)
        std::cout << qtFullPathHists.back() << std::endl;

      unsigned int qtEnabled = itQT.getParameter<unsigned int>("QualityTestSummaryEnabled");

      qtSumEnabled.push_back(qtEnabled);

      if (qtEnabled) {
        m_totalNrQtSummaryEnabled++;
      }
    }

    m_trackQualityTestName.push_back(qtNames);
    m_trackQualityTestHist.push_back(qtFullPathHists);
    m_trackQtSummaryEnabled.push_back(qtSumEnabled);

    indexSys++;
  }

  // L1 objects

  //
  m_nrHitObjects = m_hitObjects.size();

  m_hitLabel.reserve(m_nrHitObjects);
  m_hitDisable.reserve(m_nrHitObjects);

  // on average five quality test per object - just a best guess
  m_hitQualityTestName.reserve(5 * m_nrHitObjects);
  m_hitQualityTestHist.reserve(5 * m_nrHitObjects);
  m_hitQtSummaryEnabled.reserve(5 * m_nrHitObjects);

  int indexObj = 0;

  for (const auto& itObject : m_hitObjects) {
    m_hitLabel.push_back(itObject.getParameter<std::string>("HitLabel"));

    m_hitDisable.push_back(itObject.getParameter<unsigned int>("HitDisable"));
    // check the additional disable flag from m_disableHitObjects
    for (const auto& itObj : m_disableHitObjects) {
      if (itObj == m_hitLabel[indexObj]) {
        m_hitDisable[indexObj] = 1;
      }
    }

    std::vector<edm::ParameterSet> qTests = itObject.getParameter<std::vector<edm::ParameterSet> >("QualityTests");
    size_t qtPerObject = qTests.size();

    std::vector<std::string> qtNames;
    qtNames.reserve(qtPerObject);

    std::vector<std::string> qtFullPathHists;
    qtFullPathHists.reserve(qtPerObject);

    std::vector<unsigned int> qtSumEnabled;
    qtSumEnabled.reserve(qtPerObject);

    if (m_verbose)
      std::cout << "\nLooping over hit quality tests" << std::endl;
    for (const auto& itQT : qTests) {
      totalNrQualityTests++;

      qtNames.push_back(itQT.getParameter<std::string>("QualityTestName"));

      // qtFullPathHists.push_back( m_histDir + "/" + itQT.getParameter<std::string> ("QualityTestHist") );
      qtFullPathHists.push_back(itQT.getParameter<std::string>("QualityTestHist"));
      if (m_verbose)
        std::cout << qtFullPathHists.back() << std::endl;

      unsigned int qtEnabled = itQT.getParameter<unsigned int>("QualityTestSummaryEnabled");

      qtSumEnabled.push_back(qtEnabled);

      if (qtEnabled) {
        m_totalNrQtSummaryEnabled++;
      }
    }

    m_hitQualityTestName.push_back(qtNames);
    m_hitQualityTestHist.push_back(qtFullPathHists);
    m_hitQtSummaryEnabled.push_back(qtSumEnabled);

    indexObj++;
  }

  // L1 Strip Noisy=========================================================================================

  m_nrNoisyStrip = m_noisyStrip.size();

  m_noisyLabel.reserve(m_nrNoisyStrip);
  m_noisyDisable.reserve(m_nrNoisyStrip);

  // on average 20 quality tests per system
  m_noisyQualityTestName.reserve(20 * m_nrNoisyStrip);   // Not needed? - AWB 05.12.16
  m_noisyQualityTestHist.reserve(20 * m_nrNoisyStrip);   // Not needed? - AWB 05.12.16
  m_noisyQtSummaryEnabled.reserve(20 * m_nrNoisyStrip);  // Not needed? - AWB 05.12.16

  int indexNois = 0;

  for (const auto& itNoisy : m_noisyStrip) {
    m_noisyLabel.push_back(itNoisy.getParameter<std::string>("NoisyLabel"));

    m_noisyDisable.push_back(itNoisy.getParameter<unsigned int>("NoisyDisable"));
    // check the additional disable flag from m_disableNoisyObjects
    for (const auto& itNois : m_disableNoisyStrip) {
      if (itNois == m_noisyLabel[indexNois]) {
        m_noisyDisable[indexNois] = 1;
      }
    }

    std::vector<edm::ParameterSet> qTests = itNoisy.getParameter<std::vector<edm::ParameterSet> >("QualityTests");
    size_t qtPerNoisy = qTests.size();

    std::vector<std::string> qtNames;
    qtNames.reserve(qtPerNoisy);

    std::vector<std::string> qtFullPathHists;
    qtFullPathHists.reserve(qtPerNoisy);

    std::vector<unsigned int> qtSumEnabled;
    qtSumEnabled.reserve(qtPerNoisy);

    if (m_verbose)
      std::cout << "\nLooping over noisy quality tests" << std::endl;
    for (const auto& itQT : qTests) {
      totalNrQualityTests++;

      qtNames.push_back(itQT.getParameter<std::string>("QualityTestName"));

      qtFullPathHists.push_back(itQT.getParameter<std::string>("QualityTestHist"));
      if (m_verbose)
        std::cout << qtFullPathHists.back() << std::endl;

      unsigned int qtEnabled = itQT.getParameter<unsigned int>("QualityTestSummaryEnabled");

      qtSumEnabled.push_back(qtEnabled);

      if (qtEnabled) {
        m_totalNrQtSummaryEnabled++;
      }
    }

    m_noisyQualityTestName.push_back(qtNames);
    m_noisyQualityTestHist.push_back(qtFullPathHists);
    m_noisyQtSummaryEnabled.push_back(qtSumEnabled);

    indexNois++;
  }

  // L1 Strip Dead=========================================================================================

  m_nrDeadStrip = m_deadStrip.size();

  m_deadLabel.reserve(m_nrDeadStrip);
  m_deadDisable.reserve(m_nrDeadStrip);

  // on average 20 quality tests per system
  m_deadQualityTestName.reserve(20 * m_nrDeadStrip);   // Not needed? - AWB 05.12.16
  m_deadQualityTestHist.reserve(20 * m_nrDeadStrip);   // Not needed? - AWB 05.12.16
  m_deadQtSummaryEnabled.reserve(20 * m_nrDeadStrip);  // Not needed? - AWB 05.12.16

  int indexDed = 0;

  for (const auto& itDead : m_deadStrip) {
    m_deadLabel.push_back(itDead.getParameter<std::string>("DeadLabel"));

    m_deadDisable.push_back(itDead.getParameter<unsigned int>("DeadDisable"));
    // check the additional disable flag from m_disableDeadObjects
    for (const auto& itDed : m_disableDeadStrip) {
      if (itDed == m_deadLabel[indexDed]) {
        m_deadDisable[indexDed] = 1;
      }
    }

    std::vector<edm::ParameterSet> qTests = itDead.getParameter<std::vector<edm::ParameterSet> >("QualityTests");
    size_t qtPerDead = qTests.size();

    std::vector<std::string> qtNames;
    qtNames.reserve(qtPerDead);

    std::vector<std::string> qtFullPathHists;
    qtFullPathHists.reserve(qtPerDead);

    std::vector<unsigned int> qtSumEnabled;
    qtSumEnabled.reserve(qtPerDead);

    if (m_verbose)
      std::cout << "\nLooping over dead quality tests" << std::endl;
    for (const auto& itQT : qTests) {
      totalNrQualityTests++;

      qtNames.push_back(itQT.getParameter<std::string>("QualityTestName"));

      qtFullPathHists.push_back(itQT.getParameter<std::string>("QualityTestHist"));
      if (m_verbose)
        std::cout << qtFullPathHists.back() << std::endl;

      unsigned int qtEnabled = itQT.getParameter<unsigned int>("QualityTestSummaryEnabled");

      qtSumEnabled.push_back(qtEnabled);

      if (qtEnabled) {
        m_totalNrQtSummaryEnabled++;
      }
    }

    m_deadQualityTestName.push_back(qtNames);
    m_deadQualityTestHist.push_back(qtFullPathHists);
    m_deadQtSummaryEnabled.push_back(qtSumEnabled);

    indexDed++;
  }

  m_summaryContent.reserve(m_nrTrackObjects + m_nrHitObjects + m_nrNoisyStrip + m_nrDeadStrip);
  m_meReportSummaryContent.reserve(totalNrQualityTests);
}

void L1TEMTFEventInfoClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                                   DQMStore::IGetter& igetter,
                                                   const edm::LuminosityBlock& lumiSeg,
                                                   const edm::EventSetup& evSetup) {
  if (m_verbose)
    std::cout << "\nInside void L1TEMTFEventInfoClient::dqmEndLuminosityBlock" << std::endl;
  if (m_runInEndLumi) {
    book(ibooker, igetter);
    readQtResults(ibooker, igetter);

    if (m_verbose) {
      std::cout << "\n  L1TEMTFEventInfoClient::endLuminosityBlock\n" << std::endl;
      dumpContentMonitorElements(ibooker, igetter);
    }
  }
}

void L1TEMTFEventInfoClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (m_verbose)
    std::cout << "\nInside void L1TEMTFEventInfoClient::dqmEndJob" << std::endl;
  book(ibooker, igetter);

  readQtResults(ibooker, igetter);

  if (m_verbose) {
    std::cout << "\n  L1TEMTFEventInfoClient::endRun\n" << std::endl;
    dumpContentMonitorElements(ibooker, igetter);
  }
}

void L1TEMTFEventInfoClient::dumpContentMonitorElements(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (m_verbose)
    std::cout << "\nSummary report " << std::endl;

  // summary content

  MonitorElement* me = igetter.get(m_meReportSummaryMap->getName());

  if (m_verbose)
    std::cout << "\nSummary content per system and object as filled in histogram\n  " << m_meReportSummaryMap->getName()
              << std::endl;

  if (!me) {
    if (m_verbose)
      std::cout << "\nNo histogram " << m_meReportSummaryMap->getName()
                << "\nNo summary content per system and object as filled in histogram.\n  " << std::endl;
    return;
  }

  TH2F* hist = me->getTH2F();

  const int nBinsX = hist->GetNbinsX();
  const int nBinsY = hist->GetNbinsY();
  if (m_verbose)
    std::cout << nBinsX << " " << nBinsY;

  std::vector<std::vector<int> > meReportSummaryMap(nBinsX, std::vector<int>(nBinsY));

  //    for (int iBinX = 0; iBinX < nBinsX; iBinX++) {
  //        for (int iBinY = 0; iBinY < nBinsY; iBinY++) {
  //            meReportSummaryMap[iBinX][iBinY]
  //                    = static_cast<int>(me->GetBinContent(iBinX + 1, iBinY + 1));
  //        }
  //    }

  if (m_verbose)
    std::cout << "\nL1 systems: " << m_nrTrackObjects << " systems included\n"
              << "\n Summary content size: " << (m_summaryContent.size()) << std::endl;

  for (unsigned int iTrackObj = 0; iTrackObj < m_nrTrackObjects; ++iTrackObj) {
    if (m_verbose)
      std::cout << std::setw(10) << m_trackLabel[iTrackObj]
                << std::setw(10)
                // << m_trackLabelExt[iTrackObj] << " \t" << m_trackDisable[iTrackObj]
                << m_trackDisable[iTrackObj] << " \t" << std::setw(25) << " m_summaryContent[" << std::setw(2)
                << iTrackObj << "] = " << meReportSummaryMap[0][iTrackObj] << std::endl;
  }

  if (m_verbose)
    std::cout << "\n L1 trigger objects: " << m_nrHitObjects << " objects included\n" << std::endl;

  for (unsigned int iMon = m_nrTrackObjects; iMon < m_nrTrackObjects + m_nrHitObjects; ++iMon) {
    if (m_verbose)
      std::cout << std::setw(20) << m_hitLabel[iMon - m_nrTrackObjects] << " \t"
                << m_hitDisable[iMon - m_nrTrackObjects] << " \t" << std::setw(25) << " m_summaryContent["
                << std::setw(2) << iMon << "] = \t" << m_summaryContent[iMon] << std::endl;
  }

  if (m_verbose)
    std::cout << std::endl;

  // quality tests

  if (m_verbose)
    std::cout << "\nQuality test results as filled in "
              << "\n  " << m_monitorDir << "/EventInfo/reportSummaryContents\n"
              << "\n  Total number of quality tests: " << (m_meReportSummaryContent.size()) << "\n"
              << std::endl;

  for (const auto itME : m_meReportSummaryContent) {
    if (m_verbose)
      std::cout << std::setw(50) << itME->getName() << " \t" << std::setw(25) << itME->getFloatValue() << std::endl;
  }

  if (m_verbose)
    std::cout << std::endl;
}

void L1TEMTFEventInfoClient::book(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (m_verbose)
    std::cout << "\nInside void L1TEMTFEventInfoClient::book" << std::endl;
  std::string dirEventInfo = m_monitorDir + "/EventInfo";
  if (m_verbose)
    std::cout << dirEventInfo << std::endl;

  ibooker.setCurrentFolder(dirEventInfo);
  if (m_verbose)
    std::cout << "Ran ibooker.setCurrentFolder(dirEventInfo;" << std::endl;

  // ...and book it again
  m_meReportSummary = ibooker.bookFloat("reportSummary");
  if (m_verbose)
    std::cout << "Ran m_meReportSummary = ibooker.bookFloat" << std::endl;

  // initialize reportSummary to 1

  if (m_meReportSummary) {
    if (m_verbose)
      std::cout << "Initializing reportSummary to 1" << std::endl;
    m_meReportSummary->Fill(1);
    if (m_verbose)
      std::cout << "Ran m_meReportSummary->Fill(1);" << std::endl;
  }

  // define float histograms for reportSummaryContents (one histogram per quality test),
  // initialize them to zero
  // initialize also m_summaryContent to dqm::qstatus::DISABLED

  ibooker.setCurrentFolder(dirEventInfo + "/reportSummaryContents");
  if (m_verbose)
    std::cout << "Ran ibooker.setCurrentFolder(dirEventInfo" << std::endl;
  // general counters:
  //   iAllQTest: all quality tests for all systems and objects
  int iAllQTest = 0;

  if (m_verbose)
    std::cout << "m_nrTrackObjects = " << m_nrTrackObjects << std::endl;
  for (unsigned int iMon = 0; iMon < m_nrTrackObjects; ++iMon) {
    if (m_verbose)
      std::cout << "  * iMon = " << iMon << std::endl;

    m_summaryContent.push_back(dqm::qstatus::DISABLED);
    if (m_verbose)
      std::cout << "Ran m_summaryContent.push_back(dqm::qstatus::DISABLED);" << std::endl;

    const std::vector<std::string>& trackObjQtName = m_trackQualityTestName[iMon];
    if (m_verbose)
      std::cout << "Ran const std::vector<std::string>& trackObjQtName = m_trackQualityTestName[iMon];" << std::endl;

    for (const auto& itQtName : trackObjQtName) {
      if (m_verbose)
        std::cout << "    - m_monitorDir = " << m_monitorDir << ", m_trackLabel[iMon] = " << m_trackLabel[iMon]
                  << ", (itQtName) = " << (itQtName) << std::endl;

      // Avoid error in ibooker.bookFloat(hStr))
      std::string m_mon_mod = m_monitorDir;
      std::replace(m_mon_mod.begin(), m_mon_mod.end(), '/', '_');

      const std::string hStr = m_mon_mod + "_L1Sys_" + m_trackLabel[iMon] + "_" + (itQtName);
      if (m_verbose)
        std::cout << "    - " << hStr << std::endl;

      m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));
      if (m_verbose)
        std::cout << "    - Ran m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));" << std::endl;
      m_meReportSummaryContent[iAllQTest]->Fill(0.);
      if (m_verbose)
        std::cout << "    - Ran m_meReportSummaryContent[iAllQTest]->Fill(0.);" << std::endl;

      iAllQTest++;
    }
  }

  for (unsigned int iMon = 0; iMon < m_nrHitObjects; ++iMon) {
    if (m_verbose)
      std::cout << "  * iMon = " << iMon << std::endl;

    m_summaryContent.push_back(dqm::qstatus::DISABLED);

    const std::vector<std::string>& objQtName = m_hitQualityTestName[iMon];

    for (const auto& itQtName : objQtName) {
      // Avoid error in ibooker.bookFloat(hStr))
      std::string m_mon_mod = m_monitorDir;
      std::replace(m_mon_mod.begin(), m_mon_mod.end(), '/', '_');

      const std::string hStr = m_mon_mod + "_L1Obj_" + m_hitLabel[iMon] + "_" + (itQtName);
      if (m_verbose)
        std::cout << "    - " << hStr << std::endl;

      m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));
      if (m_verbose)
        std::cout << "    - Ran m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));" << std::endl;
      m_meReportSummaryContent[iAllQTest]->Fill(0.);
      if (m_verbose)
        std::cout << "    - Ran m_meReportSummaryContent[iAllQTest]->Fill(0.);" << std::endl;

      iAllQTest++;
    }
  }

  // for Noisy Strips ====================================================================
  for (unsigned int iMon = 0; iMon < m_nrNoisyStrip; ++iMon) {
    m_summaryContent.push_back(dqm::qstatus::DISABLED);

    const std::vector<std::string>& objQtName = m_noisyQualityTestName[iMon];

    for (const auto& itQtName : objQtName) {
      // Avoid error in ibooker.bookFloat(hStr))
      std::string m_mon_mod = m_monitorDir;
      std::replace(m_mon_mod.begin(), m_mon_mod.end(), '/', '_');

      const std::string hStr = m_mon_mod + "_L1Obj_" + m_noisyLabel[iMon] + "_" + (itQtName);
      if (m_verbose)
        std::cout << "    - " << hStr << std::endl;

      m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));
      m_meReportSummaryContent[iAllQTest]->Fill(0.);

      iAllQTest++;
    }
  }
  // for Dead Strips ====================================================================
  for (unsigned int iMon = 0; iMon < m_nrDeadStrip; ++iMon) {
    m_summaryContent.push_back(dqm::qstatus::DISABLED);

    const std::vector<std::string>& objQtName = m_deadQualityTestName[iMon];

    for (const auto& itQtName : objQtName) {
      // Avoid error in ibooker.bookFloat(hStr))
      std::string m_mon_mod = m_monitorDir;
      std::replace(m_mon_mod.begin(), m_mon_mod.end(), '/', '_');

      const std::string hStr = m_mon_mod + "_L1Obj_" + m_deadLabel[iMon] + "_" + (itQtName);
      if (m_verbose)
        std::cout << "    - " << hStr << std::endl;

      m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));
      m_meReportSummaryContent[iAllQTest]->Fill(0.);

      iAllQTest++;
    }
  }

  if (m_verbose)
    std::cout << "Setting current folder to " << dirEventInfo << std::endl;
  ibooker.setCurrentFolder(dirEventInfo);
  if (m_verbose)
    std::cout << "Ran ibooker.setCurrentFolder(dirEventInfo);" << std::endl;

  // define a histogram with two bins on X and maximum of m_nrTrackObjects, m_nrHitObjects on Y
  int nBinsY = std::max(m_nrTrackObjects, m_nrHitObjects);
  int nBinsYStrip = std::max(m_nrNoisyStrip, m_nrDeadStrip);

  m_meReportSummaryMap =
      ibooker.book2D("reportSummaryMap_EMTF", "reportSummaryMap_EMTF", 2, 1, 3, nBinsY, 1, nBinsY + 1);
  m_meReportSummaryMap_chamberStrip = ibooker.book2D(
      "reportSummaryMap_chamberStrip", "reportSummaryMap_chamberStrip", 2, 1, 3, nBinsYStrip, 1, nBinsYStrip + 1);

  if (m_monitorDir == "L1TEMU") {
    m_meReportSummaryMap->setTitle("L1TEMU: L1 Emulator vs Data Report Summary Map");

  } else if (m_monitorDir == "L1T") {
    m_meReportSummaryMap->setTitle("L1T: L1 Trigger Data Report Summary Map");
  } else if (m_monitorDir == "L1T2016") {
    m_meReportSummaryMap->setTitle("L1T2016: L1 Trigger Data Report Summary Map");
  } else {
    // do nothing
  }

  m_meReportSummaryMap->setAxisTitle(" ", 1);
  m_meReportSummaryMap->setAxisTitle(" ", 2);

  m_meReportSummaryMap->setBinLabel(1, "Noisy Check", 1);
  m_meReportSummaryMap->setBinLabel(2, "Dead Check", 1);

  m_meReportSummaryMap_chamberStrip->setBinLabel(1, "Noisy Check", 1);
  m_meReportSummaryMap_chamberStrip->setBinLabel(2, "Dead Check", 1);

  //    for (int iBin = 0; iBin < nBinsY; ++iBin) {
  //        m_meReportSummaryMap->setBinLabel(iBin + 1, " ", 2);   }

  m_meReportSummaryMap->setBinLabel(1, "Hit BX", 2);
  m_meReportSummaryMap->setBinLabel(2, "Track BX", 2);
  //m_meReportSummaryMap->setBinLabel(3, "Track Phi", 2);

  const std::vector<std::string> suffix_name = {"-4/2", "-4/1",  "-3/2",  "-3/1",  "-2/2",  "-2/1", "-1/3",
                                                "-1/2", "-1/1b", "-1/1a", "+1/1a", "+1/1b", "+1/2", "+1/3",
                                                "+2/1", "+2/2",  "+3/1",  "+3/2",  "+4/1",  "+4/2"};
  for (int iBin = 0; iBin < nBinsYStrip; ++iBin) {
    m_meReportSummaryMap_chamberStrip->setBinLabel(iBin + 1, "ChamberStrip " + suffix_name[iBin], 2);
  }
}

void L1TEMTFEventInfoClient::readQtResults(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  // initialize summary content, summary sum and ReportSummaryContent float histograms
  // for all L1 systems and L1 objects

  if (m_verbose)
    std::cout << "\nInside L1TEMTFEventInfoClient::readQtResults" << std::endl;  // Extra printout - AWB 03.12.16

  for (std::vector<int>::iterator it = m_summaryContent.begin(); it != m_summaryContent.end(); ++it) {
    (*it) = dqm::qstatus::DISABLED;
  }

  m_summarySum = 0.;

  for (const auto& itME : m_meReportSummaryContent) {
    itME->Fill(0.);
  }

  // general counters:
  //   iAllQTest: all quality tests for all systems and objects
  //   iAllMon:   all monitored systems and objects
  int iAllQTest = 0;
  int iAllMon = 0;

  // quality tests for all L1 systems

  for (unsigned int iTrackObj = 0; iTrackObj < m_nrTrackObjects; ++iTrackObj) {
    // get the reports for each quality test

    const std::vector<std::string>& trackObjQtName = m_trackQualityTestName[iTrackObj];
    const std::vector<std::string>& trackObjQtHist = m_trackQualityTestHist[iTrackObj];
    const std::vector<unsigned int>& trackObjQtSummaryEnabled = m_trackQtSummaryEnabled[iTrackObj];

    // pro system counter for quality tests
    int iTrackObjQTest = 0;

    for (const auto& itQtName : trackObjQtName) {
      // get results, status and message

      if (m_verbose)
        std::cout << "  itQtName = " << (itQtName) << std::endl;  // Extra printout - AWB 03.12.16

      MonitorElement* qHist = igetter.get(trackObjQtHist[iTrackObjQTest]);

      if (qHist) {
        const std::vector<QReport*> qtVec = qHist->getQReports();

        // if (m_verbose) {
        if (true) {  // Force printout - AWB 03.12.16

          if (m_verbose)
            std::cout << "  - Number of quality tests"
                      // if (m_verbose) std::cout << "\nNumber of quality tests"
                      << " for histogram " << trackObjQtHist[iTrackObjQTest] << ": " << qtVec.size() << "\n"
                      << std::endl;
        }

        const QReport* sysQReport = qHist->getQReport(itQtName);
        if (sysQReport) {
          const float trackObjQtResult = sysQReport->getQTresult();
          const int trackObjQtStatus = sysQReport->getStatus();
          const std::string& trackObjQtMessage = sysQReport->getMessage();

          if (m_verbose) {
            std::cout << "\n"
                      << (itQtName) << " quality test:"
                      << "\n  result:  " << trackObjQtResult << "\n  status:  " << trackObjQtStatus
                      << "\n  message: " << trackObjQtMessage << "\n"
                      << "\nFilling m_meReportSummaryContent[" << iAllQTest << "] with value " << trackObjQtResult
                      << "\n"
                      << std::endl;
          }

          m_meReportSummaryContent[iAllQTest]->Fill(trackObjQtResult);

          // for the summary map, keep the highest status value ("ERROR") of all tests
          // which are considered for the summary plot
          if (trackObjQtSummaryEnabled[iTrackObjQTest]) {
            if (trackObjQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = trackObjQtStatus;
            }

            m_summarySum += trackObjQtResult;
          }

        } else {
          // for the summary map, if the test was not found but it is assumed to be
          // considered for the summary plot, set it to dqm::qstatus::INVALID

          int trackObjQtStatus = dqm::qstatus::INVALID;

          if (trackObjQtSummaryEnabled[iTrackObjQTest]) {
            if (trackObjQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = trackObjQtStatus;
            }
          }

          m_meReportSummaryContent[iAllQTest]->Fill(0.);

          if (m_verbose)
            std::cout << "\n" << (itQtName) << " quality test not found\n" << std::endl;
        }

      } else {
        // for the summary map, if the histogram was not found but it is assumed
        // to have a test be considered for the summary plot, set it to dqm::qstatus::INVALID

        int trackObjQtStatus = dqm::qstatus::INVALID;

        if (trackObjQtSummaryEnabled[iTrackObjQTest]) {
          if (trackObjQtStatus > m_summaryContent[iAllMon]) {
            m_summaryContent[iAllMon] = trackObjQtStatus;
          }
        }

        m_meReportSummaryContent[iAllQTest]->Fill(0.);

        if (m_verbose)
          std::cout << "\nHistogram " << trackObjQtHist[iTrackObjQTest] << " not found\n" << std::endl;
      }

      // increase counters for quality tests
      iTrackObjQTest++;
      iAllQTest++;
    }

    iAllMon++;
  }

  // quality tests for all L1 objects

  for (unsigned int iHitObj = 0; iHitObj < m_nrHitObjects; ++iHitObj) {
    // get the reports for each quality test

    const std::vector<std::string>& hitObjQtName = m_hitQualityTestName[iHitObj];
    const std::vector<std::string>& hitObjQtHist = m_hitQualityTestHist[iHitObj];
    const std::vector<unsigned int>& hitObjQtSummaryEnabled = m_hitQtSummaryEnabled[iHitObj];

    // pro object counter for quality tests
    int iHitObjQTest = 0;

    for (const auto& itQtName : hitObjQtName) {
      // get results, status and message

      MonitorElement* qHist = igetter.get(hitObjQtHist[iHitObjQTest]);

      if (qHist) {
        const std::vector<QReport*> qtVec = qHist->getQReports();

        if (m_verbose)
          std::cout << "\nNumber of quality tests "
                    << " for histogram " << hitObjQtHist[iHitObjQTest] << ": " << qtVec.size() << "\n"
                    << std::endl;

        const QReport* objQReport = qHist->getQReport(itQtName);
        if (objQReport) {
          const float hitObjQtResult = objQReport->getQTresult();
          const int hitObjQtStatus = objQReport->getStatus();
          const std::string& hitObjQtMessage = objQReport->getMessage();

          if (m_verbose) {
            std::cout << "\n"
                      << (itQtName) << " quality test:"
                      << "\n  result:  " << hitObjQtResult << "\n  status:  " << hitObjQtStatus
                      << "\n  message: " << hitObjQtMessage << "\n"
                      << "\nFilling m_meReportSummaryContent[" << iAllQTest << "] with value " << hitObjQtResult << "\n"
                      << std::endl;
          }

          m_meReportSummaryContent[iAllQTest]->Fill(hitObjQtResult);

          // for the summary map, keep the highest status value ("ERROR") of all tests
          // which are considered for the summary plot
          if (hitObjQtSummaryEnabled[iHitObjQTest]) {
            if (hitObjQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = hitObjQtStatus;
            }

            m_summarySum += hitObjQtResult;
          }

        } else {
          // for the summary map, if the test was not found but it is assumed to be
          // considered for the summary plot, set it to dqm::qstatus::INVALID

          int hitObjQtStatus = dqm::qstatus::INVALID;

          if (hitObjQtSummaryEnabled[iHitObjQTest]) {
            if (hitObjQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = hitObjQtStatus;
            }
          }

          m_meReportSummaryContent[iAllQTest]->Fill(0.);

          if (m_verbose)
            std::cout << "\n" << (itQtName) << " quality test not found\n" << std::endl;
        }

      } else {
        // for the summary map, if the histogram was not found but it is assumed
        // to have a test be considered for the summary plot, set it to dqm::qstatus::INVALID

        int hitObjQtStatus = dqm::qstatus::INVALID;

        if (hitObjQtSummaryEnabled[iHitObjQTest]) {
          if (hitObjQtStatus > m_summaryContent[iAllMon]) {
            m_summaryContent[iAllMon] = hitObjQtStatus;
          }
        }

        m_meReportSummaryContent[iAllQTest]->Fill(0.);

        if (m_verbose)
          std::cout << "\nHistogram " << hitObjQtHist[iHitObjQTest] << " not found\n" << std::endl;
      }
      // increase counters for quality tests
      iHitObjQTest++;
      iAllQTest++;
    }
    iAllMon++;
  }

  // quality tests for all L1 Noisy Strip =================================================================

  for (unsigned int iNoisyStrp = 0; iNoisyStrp < m_nrNoisyStrip; ++iNoisyStrp) {
    // get the reports for each quality test
    const std::vector<std::string>& noisyStrpQtName = m_noisyQualityTestName[iNoisyStrp];
    const std::vector<std::string>& noisyStrpQtHist = m_noisyQualityTestHist[iNoisyStrp];
    const std::vector<unsigned int>& noisyStrpQtSummaryEnabled = m_noisyQtSummaryEnabled[iNoisyStrp];

    // pro object counter for quality tests
    int iNoisyStrpQTest = 0;

    for (const auto& itQtName : noisyStrpQtName) {
      // get results, status and message
      MonitorElement* qHist = igetter.get(noisyStrpQtHist[iNoisyStrpQTest]);

      if (qHist) {
        const std::vector<QReport*> qtVec = qHist->getQReports();

        if (m_verbose)
          std::cout << "\nNumber of quality tests "
                    << " for histogram " << noisyStrpQtHist[iNoisyStrpQTest] << ": " << qtVec.size() << "\n"
                    << std::endl;

        const QReport* objQReport = qHist->getQReport(itQtName);
        if (objQReport) {
          const float noisyStrpQtResult = objQReport->getQTresult();
          const int noisyStrpQtStatus = objQReport->getStatus();
          const std::string& noisyStrpQtMessage = objQReport->getMessage();

          if (m_verbose) {
            std::cout << "\n"
                      << (itQtName) << " quality test:"
                      << "\n  result:  " << noisyStrpQtResult << "\n  status:  " << noisyStrpQtStatus
                      << "\n  message: " << noisyStrpQtMessage << "\n"
                      << "\nFilling m_meReportSummaryContent[" << iAllQTest << "] with value " << noisyStrpQtResult
                      << "\n"
                      << std::endl;
          }

          m_meReportSummaryContent[iAllQTest]->Fill(noisyStrpQtResult);

          // for the summary map, keep the highest status value ("ERROR") of all tests
          // which are considered for the summary plot
          if (noisyStrpQtSummaryEnabled[iNoisyStrpQTest]) {
            if (noisyStrpQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = noisyStrpQtStatus;
            }
            m_summarySum += noisyStrpQtResult;
          }

        } else {
          // for the summary map, if the test was not found but it is assumed to be
          // considered for the summary plot, set it to dqm::qstatus::INVALID

          int noisyStrpQtStatus = dqm::qstatus::INVALID;

          if (noisyStrpQtSummaryEnabled[iNoisyStrpQTest]) {
            if (noisyStrpQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = noisyStrpQtStatus;
            }
          }

          m_meReportSummaryContent[iAllQTest]->Fill(0.);

          if (m_verbose)
            std::cout << "\n" << (itQtName) << " quality test not found\n" << std::endl;
        }

      } else {
        // for the summary map, if the histogram was not found but it is assumed
        // to have a test be considered for the summary plot, set it to dqm::qstatus::INVALID

        int noisyStrpQtStatus = dqm::qstatus::INVALID;

        if (noisyStrpQtSummaryEnabled[iNoisyStrpQTest]) {
          if (noisyStrpQtStatus > m_summaryContent[iAllMon]) {
            m_summaryContent[iAllMon] = noisyStrpQtStatus;
          }
        }

        m_meReportSummaryContent[iAllQTest]->Fill(0.);

        if (m_verbose)
          std::cout << "\nHistogram " << noisyStrpQtHist[iNoisyStrpQTest] << " not found\n" << std::endl;
      }
      // increase counters for quality tests
      iNoisyStrpQTest++;
      iAllQTest++;
    }
    iAllMon++;
  }
  // quality tests for all L1 Dead Strip =================================================================

  for (unsigned int iDeadStrp = 0; iDeadStrp < m_nrDeadStrip; ++iDeadStrp) {
    // get the reports for each quality test
    const std::vector<std::string>& deadStrpQtName = m_deadQualityTestName[iDeadStrp];
    const std::vector<std::string>& deadStrpQtHist = m_deadQualityTestHist[iDeadStrp];
    const std::vector<unsigned int>& deadStrpQtSummaryEnabled = m_deadQtSummaryEnabled[iDeadStrp];

    // pro object counter for quality tests
    int iDeadStrpQTest = 0;

    for (const auto& itQtName : deadStrpQtName) {
      // get results, status and message

      MonitorElement* qHist = igetter.get(deadStrpQtHist[iDeadStrpQTest]);

      if (qHist) {
        const std::vector<QReport*> qtVec = qHist->getQReports();

        if (m_verbose)
          std::cout << "\nNumber of quality tests "
                    << " for histogram " << deadStrpQtHist[iDeadStrpQTest] << ": " << qtVec.size() << "\n"
                    << std::endl;

        const QReport* objQReport = qHist->getQReport(itQtName);
        if (objQReport) {
          const float deadStrpQtResult = objQReport->getQTresult();
          const int deadStrpQtStatus = objQReport->getStatus();
          const std::string& deadStrpQtMessage = objQReport->getMessage();

          if (m_verbose) {
            std::cout << "\n"
                      << (itQtName) << " quality test:"
                      << "\n  result:  " << deadStrpQtResult << "\n  status:  " << deadStrpQtStatus
                      << "\n  message: " << deadStrpQtMessage << "\n"
                      << "\nFilling m_meReportSummaryContent[" << iAllQTest << "] with value " << deadStrpQtResult
                      << "\n"
                      << std::endl;
          }

          m_meReportSummaryContent[iAllQTest]->Fill(deadStrpQtResult);

          // for the summary map, keep the highest status value ("ERROR") of all tests
          // which are considered for the summary plot
          if (deadStrpQtSummaryEnabled[iDeadStrpQTest]) {
            if (deadStrpQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = deadStrpQtStatus;
            }
            m_summarySum += deadStrpQtResult;
          }

        } else {
          // for the summary map, if the test was not found but it is assumed to be
          // considered for the summary plot, set it to dqm::qstatus::INVALID

          int deadStrpQtStatus = dqm::qstatus::INVALID;

          if (deadStrpQtSummaryEnabled[iDeadStrpQTest]) {
            if (deadStrpQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = deadStrpQtStatus;
            }
          }

          m_meReportSummaryContent[iAllQTest]->Fill(0.);

          if (m_verbose)
            std::cout << "\n" << (itQtName) << " quality test not found\n" << std::endl;
        }

      } else {
        // for the summary map, if the histogram was not found but it is assumed
        // to have a test be considered for the summary plot, set it to dqm::qstatus::INVALID

        int deadStrpQtStatus = dqm::qstatus::INVALID;

        if (deadStrpQtSummaryEnabled[iDeadStrpQTest]) {
          if (deadStrpQtStatus > m_summaryContent[iAllMon]) {
            m_summaryContent[iAllMon] = deadStrpQtStatus;
          }
        }

        m_meReportSummaryContent[iAllQTest]->Fill(0.);

        if (m_verbose)
          std::cout << "\nHistogram " << deadStrpQtHist[iDeadStrpQTest] << " not found\n" << std::endl;
      }

      // increase counters for quality tests
      iDeadStrpQTest++;
      iAllQTest++;
    }
    iAllMon++;
  }

  // reportSummary value
  m_reportSummary = m_summarySum / float(m_totalNrQtSummaryEnabled);
  if (m_meReportSummary) {
    m_meReportSummary->Fill(m_reportSummary);
  }

  // fill the ReportSummaryMap for L1 systems (bin 1 on X)
  for (unsigned int iTrackObj = 0; iTrackObj < m_nrTrackObjects; ++iTrackObj) {
    double summCont = static_cast<double>(m_summaryContent[iTrackObj]);
    m_meReportSummaryMap->setBinContent(1, iTrackObj + 1, summCont);
  }
  // fill the ReportSummaryMap for L1 objects (bin 2 on X)
  for (unsigned int iMon = m_nrTrackObjects; iMon < m_nrTrackObjects + m_nrHitObjects; ++iMon) {
    double summCont = static_cast<double>(m_summaryContent[iMon]);
    m_meReportSummaryMap->setBinContent(2, iMon - m_nrTrackObjects + 1, summCont);
  }

  // fill the ReportSummaryMap_chamberStrip for L1 Noisy Strip (bin 1 on X)
  for (unsigned int iNoisyStrp = m_nrTrackObjects + m_nrHitObjects;
       iNoisyStrp < m_nrTrackObjects + m_nrHitObjects + m_nrNoisyStrip;
       ++iNoisyStrp) {
    double summCont = static_cast<double>(m_summaryContent[iNoisyStrp]);
    m_meReportSummaryMap_chamberStrip->setBinContent(1, iNoisyStrp - m_nrTrackObjects - m_nrHitObjects + 1, summCont);
  }
  // fill the ReportSummaryMap_chamberStrip for L1 objects (bin 2 on X)
  for (unsigned int iDeadStrp = m_nrTrackObjects + m_nrHitObjects + m_nrNoisyStrip;
       iDeadStrp < m_nrTrackObjects + m_nrHitObjects + m_nrNoisyStrip + m_nrDeadStrip;
       ++iDeadStrp) {
    double summCont = static_cast<double>(m_summaryContent[iDeadStrp]);
    m_meReportSummaryMap_chamberStrip->setBinContent(
        2, iDeadStrp - m_nrTrackObjects - m_nrHitObjects - m_nrNoisyStrip + 1, summCont);
  }
}
