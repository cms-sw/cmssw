/**
 * \class L1TEventInfoClient
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
#include "DQM/L1TMonitorClient/interface/L1TEventInfoClient.h"

// system include files
#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <cmath>
#include <memory>

#include <vector>
#include <string>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <TH2F.h>
#include "TROOT.h"

// constructor
L1TEventInfoClient::L1TEventInfoClient(const edm::ParameterSet& parSet)
    : m_verbose(parSet.getUntrackedParameter<bool>("verbose", false)),
      m_monitorDir(parSet.getUntrackedParameter<std::string>("monitorDir", "")),
      m_runInEventLoop(parSet.getUntrackedParameter<bool>("runInEventLoop", false)),
      m_runInEndLumi(parSet.getUntrackedParameter<bool>("runInEndLumi", false)),
      m_runInEndRun(parSet.getUntrackedParameter<bool>("runInEndRun", false)),
      m_runInEndJob(parSet.getUntrackedParameter<bool>("runInEndJob", false)),
      m_l1Systems(parSet.getParameter<std::vector<edm::ParameterSet> >("L1Systems")),
      m_l1Objects(parSet.getParameter<std::vector<edm::ParameterSet> >("L1Objects")),
      m_disableL1Systems(parSet.getParameter<std::vector<std::string> >("DisableL1Systems")),
      m_disableL1Objects(parSet.getParameter<std::vector<std::string> >("DisableL1Objects")),
      m_nrL1Systems(0),
      m_nrL1Objects(0),
      m_totalNrQtSummaryEnabled(0) {
  initialize();
}

// destructor
L1TEventInfoClient::~L1TEventInfoClient() {
  //empty
}

void L1TEventInfoClient::initialize() {
  if (m_verbose) {
    std::cout << "\nMonitor directory =             " << m_monitorDir << std::endl;
  }

  // L1 systems

  m_nrL1Systems = m_l1Systems.size();

  m_systemLabel.reserve(m_nrL1Systems);
  m_systemLabelExt.reserve(m_nrL1Systems);
  m_systemDisable.reserve(m_nrL1Systems);

  // on average five quality test per system - just a best guess
  m_systemQualityTestName.reserve(5 * m_nrL1Systems);
  m_systemQualityTestHist.reserve(5 * m_nrL1Systems);
  m_systemQtSummaryEnabled.reserve(5 * m_nrL1Systems);

  int indexSys = 0;

  int totalNrQualityTests = 0;

  for (std::vector<edm::ParameterSet>::const_iterator itSystem = m_l1Systems.begin(); itSystem != m_l1Systems.end();
       ++itSystem) {
    m_systemLabel.push_back(itSystem->getParameter<std::string>("SystemLabel"));

    m_systemLabelExt.push_back(itSystem->getParameter<std::string>("HwValLabel"));

    m_systemDisable.push_back(itSystem->getParameter<unsigned int>("SystemDisable"));
    // check the additional disable flag from m_disableL1Systems
    for (std::vector<std::string>::const_iterator itSys = m_disableL1Systems.begin(); itSys != m_disableL1Systems.end();
         ++itSys) {
      if (*itSys == m_systemLabel[indexSys]) {
        m_systemDisable[indexSys] = 1;
      }
    }

    std::vector<edm::ParameterSet> qTests = itSystem->getParameter<std::vector<edm::ParameterSet> >("QualityTests");
    size_t qtPerSystem = qTests.size();

    std::vector<std::string> qtNames;
    qtNames.reserve(qtPerSystem);

    std::vector<std::string> qtFullPathHists;
    qtFullPathHists.reserve(qtPerSystem);

    std::vector<unsigned int> qtSumEnabled;
    qtSumEnabled.reserve(qtPerSystem);

    for (std::vector<edm::ParameterSet>::const_iterator itQT = qTests.begin(); itQT != qTests.end(); ++itQT) {
      totalNrQualityTests++;

      qtNames.push_back(itQT->getParameter<std::string>("QualityTestName"));

      qtFullPathHists.push_back(itQT->getParameter<std::string>("QualityTestHist"));

      unsigned int qtEnabled = itQT->getParameter<unsigned int>("QualityTestSummaryEnabled");

      qtSumEnabled.push_back(qtEnabled);

      if (qtEnabled) {
        m_totalNrQtSummaryEnabled++;
      }
    }

    m_systemQualityTestName.push_back(qtNames);
    m_systemQualityTestHist.push_back(qtFullPathHists);
    m_systemQtSummaryEnabled.push_back(qtSumEnabled);

    indexSys++;
  }

  // L1 objects

  //
  m_nrL1Objects = m_l1Objects.size();

  m_objectLabel.reserve(m_nrL1Objects);
  m_objectDisable.reserve(m_nrL1Objects);

  // on average five quality test per object - just a best guess
  m_objectQualityTestName.reserve(5 * m_nrL1Objects);
  m_objectQualityTestHist.reserve(5 * m_nrL1Objects);
  m_objectQtSummaryEnabled.reserve(5 * m_nrL1Objects);

  int indexObj = 0;

  for (std::vector<edm::ParameterSet>::const_iterator itObject = m_l1Objects.begin(); itObject != m_l1Objects.end();
       ++itObject) {
    m_objectLabel.push_back(itObject->getParameter<std::string>("ObjectLabel"));

    m_objectDisable.push_back(itObject->getParameter<unsigned int>("ObjectDisable"));
    // check the additional disable flag from m_disableL1Objects
    for (std::vector<std::string>::const_iterator itObj = m_disableL1Objects.begin(); itObj != m_disableL1Objects.end();
         ++itObj) {
      if (*itObj == m_objectLabel[indexObj]) {
        m_objectDisable[indexObj] = 1;
      }
    }

    std::vector<edm::ParameterSet> qTests = itObject->getParameter<std::vector<edm::ParameterSet> >("QualityTests");
    size_t qtPerObject = qTests.size();

    std::vector<std::string> qtNames;
    qtNames.reserve(qtPerObject);

    std::vector<std::string> qtFullPathHists;
    qtFullPathHists.reserve(qtPerObject);

    std::vector<unsigned int> qtSumEnabled;
    qtSumEnabled.reserve(qtPerObject);

    for (std::vector<edm::ParameterSet>::const_iterator itQT = qTests.begin(); itQT != qTests.end(); ++itQT) {
      totalNrQualityTests++;

      qtNames.push_back(itQT->getParameter<std::string>("QualityTestName"));

      qtFullPathHists.push_back(itQT->getParameter<std::string>("QualityTestHist"));

      unsigned int qtEnabled = itQT->getParameter<unsigned int>("QualityTestSummaryEnabled");

      qtSumEnabled.push_back(qtEnabled);

      if (qtEnabled) {
        m_totalNrQtSummaryEnabled++;
      }
    }

    m_objectQualityTestName.push_back(qtNames);
    m_objectQualityTestHist.push_back(qtFullPathHists);
    m_objectQtSummaryEnabled.push_back(qtSumEnabled);

    indexObj++;
  }

  m_summaryContent.reserve(m_nrL1Systems + m_nrL1Objects);
  m_meReportSummaryContent.reserve(totalNrQualityTests);
}

void L1TEventInfoClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                               DQMStore::IGetter& igetter,
                                               const edm::LuminosityBlock& lumiSeg,
                                               const edm::EventSetup& evSetup) {
  if (m_runInEndLumi) {
    book(ibooker, igetter);
    readQtResults(ibooker, igetter);

    if (m_verbose) {
      std::cout << "\n  L1TEventInfoClient::endLuminosityBlock\n" << std::endl;
      dumpContentMonitorElements(ibooker, igetter);
    }
  }
}

void L1TEventInfoClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  book(ibooker, igetter);

  readQtResults(ibooker, igetter);

  if (m_verbose) {
    std::cout << "\n  L1TEventInfoClient::endRun\n" << std::endl;
    dumpContentMonitorElements(ibooker, igetter);
  }
}

void L1TEventInfoClient::dumpContentMonitorElements(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  std::cout << "\nSummary report " << std::endl;

  // summary content

  MonitorElement* me = igetter.get(m_meReportSummaryMap->getName());

  std::cout << "\nSummary content per system and object as filled in histogram\n  " << m_meReportSummaryMap->getName()
            << std::endl;

  if (!me) {
    std::cout << "\nNo histogram " << m_meReportSummaryMap->getName()
              << "\nNo summary content per system and object as filled in histogram.\n  " << std::endl;
    return;
  }

  TH2F* hist = me->getTH2F();

  const int nBinsX = hist->GetNbinsX();
  const int nBinsY = hist->GetNbinsY();
  std::cout << nBinsX << " " << nBinsY;

  std::vector<std::vector<int> > meReportSummaryMap(nBinsX, std::vector<int>(nBinsY));

  //    for (int iBinX = 0; iBinX < nBinsX; iBinX++) {
  //        for (int iBinY = 0; iBinY < nBinsY; iBinY++) {
  //            meReportSummaryMap[iBinX][iBinY]
  //                    = static_cast<int>(me->GetBinContent(iBinX + 1, iBinY + 1));
  //        }
  //    }

  std::cout << "\nL1 systems: " << m_nrL1Systems << " systems included\n"
            << "\n Summary content size: " << (m_summaryContent.size()) << std::endl;

  for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
    std::cout << std::setw(10) << m_systemLabel[iSys] << std::setw(10) << m_systemLabelExt[iSys] << " \t"
              << m_systemDisable[iSys] << " \t" << std::setw(25) << " m_summaryContent[" << std::setw(2) << iSys
              << "] = " << meReportSummaryMap[0][iSys] << std::endl;
  }

  std::cout << "\n L1 trigger objects: " << m_nrL1Objects << " objects included\n" << std::endl;

  for (unsigned int iMon = m_nrL1Systems; iMon < m_nrL1Systems + m_nrL1Objects; ++iMon) {
    std::cout << std::setw(20) << m_objectLabel[iMon - m_nrL1Systems] << " \t" << m_objectDisable[iMon - m_nrL1Systems]
              << " \t" << std::setw(25) << " m_summaryContent[" << std::setw(2) << iMon << "] = \t"
              << m_summaryContent[iMon] << std::endl;
  }

  std::cout << std::endl;

  // quality tests

  std::cout << "\nQuality test results as filled in "
            << "\n  " << m_monitorDir << "/EventInfo/reportSummaryContents\n"
            << "\n  Total number of quality tests: " << (m_meReportSummaryContent.size()) << "\n"
            << std::endl;

  for (std::vector<MonitorElement*>::const_iterator itME = m_meReportSummaryContent.begin();
       itME != m_meReportSummaryContent.end();
       ++itME) {
    std::cout << std::setw(50) << (*itME)->getName() << " \t" << std::setw(25) << (*itME)->getFloatValue() << std::endl;
  }

  std::cout << std::endl;
}

void L1TEventInfoClient::book(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  std::string dirEventInfo = m_monitorDir + "/EventInfo";

  ibooker.setCurrentFolder(dirEventInfo);

  // ...and book it again
  m_meReportSummary = ibooker.bookFloat("reportSummary");

  // initialize reportSummary to 1

  if (m_meReportSummary) {
    m_meReportSummary->Fill(1);
  }

  // define float histograms for reportSummaryContents (one histogram per quality test),
  // initialize them to zero
  // initialize also m_summaryContent to dqm::qstatus::DISABLED

  ibooker.setCurrentFolder(dirEventInfo + "/reportSummaryContents");
  // general counters:
  //   iAllQTest: all quality tests for all systems and objects
  int iAllQTest = 0;

  for (unsigned int iMon = 0; iMon < m_nrL1Systems; ++iMon) {
    m_summaryContent.push_back(dqm::qstatus::DISABLED);

    const std::vector<std::string>& sysQtName = m_systemQualityTestName[iMon];

    for (std::vector<std::string>::const_iterator itQtName = sysQtName.begin(); itQtName != sysQtName.end();
         ++itQtName) {
      const std::string hStr = m_monitorDir + "_L1Sys_" + m_systemLabel[iMon] + "_" + (*itQtName);

      m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));
      m_meReportSummaryContent[iAllQTest]->Fill(0.);

      iAllQTest++;
    }
  }

  for (unsigned int iMon = 0; iMon < m_nrL1Objects; ++iMon) {
    m_summaryContent.push_back(dqm::qstatus::DISABLED);

    const std::vector<std::string>& objQtName = m_objectQualityTestName[iMon];

    for (std::vector<std::string>::const_iterator itQtName = objQtName.begin(); itQtName != objQtName.end();
         ++itQtName) {
      const std::string hStr = m_monitorDir + "_L1Obj_" + m_objectLabel[iMon] + "_" + (*itQtName);

      m_meReportSummaryContent.push_back(ibooker.bookFloat(hStr));
      m_meReportSummaryContent[iAllQTest]->Fill(0.);

      iAllQTest++;
    }
  }

  ibooker.setCurrentFolder(dirEventInfo);

  // define a histogram with two bins on X and maximum of m_nrL1Systems, m_nrL1Objects on Y

  int nBinsY = std::max(m_nrL1Systems, m_nrL1Objects);

  m_meReportSummaryMap = ibooker.book2D("reportSummaryMap", "reportSummaryMap", 2, 1, 3, nBinsY, 1, nBinsY + 1);

  if (m_monitorDir == "L1TEMU") {
    m_meReportSummaryMap->setTitle("L1TEMU: L1 Emulator vs Data Report Summary Map");

  } else if (m_monitorDir == "L1T") {
    m_meReportSummaryMap->setTitle("L1T: L1 Trigger Data Report Summary Map");
  } else {
    // do nothing
  }

  m_meReportSummaryMap->setAxisTitle("", 1);
  m_meReportSummaryMap->setAxisTitle("", 2);

  m_meReportSummaryMap->setBinLabel(1, "L1 systems", 1);
  m_meReportSummaryMap->setBinLabel(2, "L1 objects", 1);

  for (int iBin = 0; iBin < nBinsY; ++iBin) {
    m_meReportSummaryMap->setBinLabel(iBin + 1, " ", 2);
  }
}

void L1TEventInfoClient::readQtResults(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  // initialize summary content, summary sum and ReportSummaryContent float histograms
  // for all L1 systems and L1 objects

  for (std::vector<int>::iterator it = m_summaryContent.begin(); it != m_summaryContent.end(); ++it) {
    (*it) = dqm::qstatus::DISABLED;
  }

  m_summarySum = 0.;

  for (std::vector<MonitorElement*>::iterator itME = m_meReportSummaryContent.begin();
       itME != m_meReportSummaryContent.end();
       ++itME) {
    (*itME)->Fill(0.);
  }

  // general counters:
  //   iAllQTest: all quality tests for all systems and objects
  //   iAllMon:   all monitored systems and objects
  int iAllQTest = 0;
  int iAllMon = 0;

  // quality tests for all L1 systems

  for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
    // get the reports for each quality test

    const std::vector<std::string>& sysQtName = m_systemQualityTestName[iSys];
    const std::vector<std::string>& sysQtHist = m_systemQualityTestHist[iSys];
    const std::vector<unsigned int>& sysQtSummaryEnabled = m_systemQtSummaryEnabled[iSys];

    // pro system counter for quality tests
    int iSysQTest = 0;

    for (std::vector<std::string>::const_iterator itQtName = sysQtName.begin(); itQtName != sysQtName.end();
         ++itQtName) {
      // get results, status and message

      MonitorElement* qHist = igetter.get(sysQtHist[iSysQTest]);

      if (qHist) {
        const std::vector<QReport*> qtVec = qHist->getQReports();

        if (m_verbose) {
          std::cout << "\nNumber of quality tests "
                    << " for histogram " << sysQtHist[iSysQTest] << ": " << qtVec.size() << "\n"
                    << std::endl;
        }

        const QReport* sysQReport = qHist->getQReport(*itQtName);
        if (sysQReport) {
          const float sysQtResult = sysQReport->getQTresult();
          const int sysQtStatus = sysQReport->getStatus();
          const std::string& sysQtMessage = sysQReport->getMessage();

          if (m_verbose) {
            std::cout << "\n"
                      << (*itQtName) << " quality test:"
                      << "\n  result:  " << sysQtResult << "\n  status:  " << sysQtStatus
                      << "\n  message: " << sysQtMessage << "\n"
                      << "\nFilling m_meReportSummaryContent[" << iAllQTest << "] with value " << sysQtResult << "\n"
                      << std::endl;
          }

          m_meReportSummaryContent[iAllQTest]->Fill(sysQtResult);

          // for the summary map, keep the highest status value ("ERROR") of all tests
          // which are considered for the summary plot
          if (sysQtSummaryEnabled[iSysQTest]) {
            if (sysQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = sysQtStatus;
            }

            m_summarySum += sysQtResult;
          }

        } else {
          // for the summary map, if the test was not found but it is assumed to be
          // considered for the summary plot, set it to dqm::qstatus::INVALID

          int sysQtStatus = dqm::qstatus::INVALID;

          if (sysQtSummaryEnabled[iSysQTest]) {
            if (sysQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = sysQtStatus;
            }
          }

          m_meReportSummaryContent[iAllQTest]->Fill(0.);

          if (m_verbose) {
            std::cout << "\n" << (*itQtName) << " quality test not found\n" << std::endl;
          }
        }

      } else {
        // for the summary map, if the histogram was not found but it is assumed
        // to have a test be considered for the summary plot, set it to dqm::qstatus::INVALID

        int sysQtStatus = dqm::qstatus::INVALID;

        if (sysQtSummaryEnabled[iSysQTest]) {
          if (sysQtStatus > m_summaryContent[iAllMon]) {
            m_summaryContent[iAllMon] = sysQtStatus;
          }
        }

        m_meReportSummaryContent[iAllQTest]->Fill(0.);

        if (m_verbose) {
          std::cout << "\nHistogram " << sysQtHist[iSysQTest] << " not found\n" << std::endl;
        }
      }

      // increase counters for quality tests
      iSysQTest++;
      iAllQTest++;
    }

    iAllMon++;
  }

  // quality tests for all L1 objects

  for (unsigned int iObj = 0; iObj < m_nrL1Objects; ++iObj) {
    // get the reports for each quality test

    const std::vector<std::string>& objQtName = m_objectQualityTestName[iObj];
    const std::vector<std::string>& objQtHist = m_objectQualityTestHist[iObj];
    const std::vector<unsigned int>& objQtSummaryEnabled = m_objectQtSummaryEnabled[iObj];

    // pro object counter for quality tests
    int iObjQTest = 0;

    for (std::vector<std::string>::const_iterator itQtName = objQtName.begin(); itQtName != objQtName.end();
         ++itQtName) {
      // get results, status and message

      MonitorElement* qHist = igetter.get(objQtHist[iObjQTest]);

      if (qHist) {
        const std::vector<QReport*> qtVec = qHist->getQReports();

        if (m_verbose) {
          std::cout << "\nNumber of quality tests "
                    << " for histogram " << objQtHist[iObjQTest] << ": " << qtVec.size() << "\n"
                    << std::endl;
        }

        const QReport* objQReport = qHist->getQReport(*itQtName);
        if (objQReport) {
          const float objQtResult = objQReport->getQTresult();
          const int objQtStatus = objQReport->getStatus();
          const std::string& objQtMessage = objQReport->getMessage();

          if (m_verbose) {
            std::cout << "\n"
                      << (*itQtName) << " quality test:"
                      << "\n  result:  " << objQtResult << "\n  status:  " << objQtStatus
                      << "\n  message: " << objQtMessage << "\n"
                      << "\nFilling m_meReportSummaryContent[" << iAllQTest << "] with value " << objQtResult << "\n"
                      << std::endl;
          }

          m_meReportSummaryContent[iAllQTest]->Fill(objQtResult);

          // for the summary map, keep the highest status value ("ERROR") of all tests
          // which are considered for the summary plot
          if (objQtSummaryEnabled[iObjQTest]) {
            if (objQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = objQtStatus;
            }

            m_summarySum += objQtResult;
          }

        } else {
          // for the summary map, if the test was not found but it is assumed to be
          // considered for the summary plot, set it to dqm::qstatus::INVALID

          int objQtStatus = dqm::qstatus::INVALID;

          if (objQtSummaryEnabled[iObjQTest]) {
            if (objQtStatus > m_summaryContent[iAllMon]) {
              m_summaryContent[iAllMon] = objQtStatus;
            }
          }

          m_meReportSummaryContent[iAllQTest]->Fill(0.);

          if (m_verbose) {
            std::cout << "\n" << (*itQtName) << " quality test not found\n" << std::endl;
          }
        }

      } else {
        // for the summary map, if the histogram was not found but it is assumed
        // to have a test be considered for the summary plot, set it to dqm::qstatus::INVALID

        int objQtStatus = dqm::qstatus::INVALID;

        if (objQtSummaryEnabled[iObjQTest]) {
          if (objQtStatus > m_summaryContent[iAllMon]) {
            m_summaryContent[iAllMon] = objQtStatus;
          }
        }

        m_meReportSummaryContent[iAllQTest]->Fill(0.);

        if (m_verbose) {
          std::cout << "\nHistogram " << objQtHist[iObjQTest] << " not found\n" << std::endl;
        }
      }

      // increase counters for quality tests
      iObjQTest++;
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
  for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
    double summCont = static_cast<double>(m_summaryContent[iSys]);
    m_meReportSummaryMap->setBinContent(1, iSys + 1, summCont);
  }

  // fill the ReportSummaryMap for L1 objects (bin 2 on X)
  for (unsigned int iMon = m_nrL1Systems; iMon < m_nrL1Systems + m_nrL1Objects; ++iMon) {
    double summCont = static_cast<double>(m_summaryContent[iMon]);
    m_meReportSummaryMap->setBinContent(2, iMon - m_nrL1Systems + 1, summCont);
  }
}
