#include "DQM/L1TMonitorClient/interface/L1EmulatorErrorFlagClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "TRandom.h"
#include <TF1.h>
#include <cstdio>
#include <sstream>
#include <cmath>
#include <TProfile.h>
#include <TProfile2D.h>
#include <memory>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <fstream>
#include "TROOT.h"

L1EmulatorErrorFlagClient::L1EmulatorErrorFlagClient(const edm::ParameterSet& parSet)
    : m_verbose(parSet.getUntrackedParameter<bool>("verbose", false)),
      m_l1Systems(parSet.getParameter<std::vector<edm::ParameterSet> >("L1Systems")),
      m_nrL1Systems(0) {
  initialize();
}

L1EmulatorErrorFlagClient::~L1EmulatorErrorFlagClient() {
  //empty
}

void L1EmulatorErrorFlagClient::initialize() {
  m_nrL1Systems = m_l1Systems.size();

  m_systemLabel.reserve(m_nrL1Systems);
  m_systemLabelExt.reserve(m_nrL1Systems);
  m_systemMask.reserve(m_nrL1Systems);
  m_systemFolder.reserve(m_nrL1Systems);
  m_systemErrorFlag.reserve(m_nrL1Systems);

  int indexSys = 0;

  for (std::vector<edm::ParameterSet>::const_iterator itSystem = m_l1Systems.begin(); itSystem != m_l1Systems.end();
       ++itSystem) {
    m_systemLabel.push_back(itSystem->getParameter<std::string>("SystemLabel"));

    m_systemLabelExt.push_back(itSystem->getParameter<std::string>("HwValLabel"));

    m_systemMask.push_back(itSystem->getParameter<unsigned int>("SystemMask"));

    m_systemFolder.push_back(itSystem->getParameter<std::string>("SystemFolder"));

    indexSys++;
  }

  // [SYS]ErrorFlag histogram
  for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
    if (m_systemFolder[iSys].empty()) {
      m_systemErrorFlag.push_back("L1TEMU/" + m_systemLabel[iSys] + "/" + m_systemLabelExt[iSys] + "ErrorFlag");
    } else {
      m_systemErrorFlag.push_back(m_systemFolder[iSys] + "/" + m_systemLabelExt[iSys] + "ErrorFlag");
    }
  }

  m_summaryContent.reserve(m_nrL1Systems);
}

void L1EmulatorErrorFlagClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  ibooker.setCurrentFolder("L1TEMU/EventInfo");

  // define a histogram
  m_meSummaryErrorFlagMap =
      ibooker.book1D("L1SummaryErrorFlagMap", "L1SummaryErrorFlagMap", m_nrL1Systems, 1, m_nrL1Systems + 1);

  m_meSummaryErrorFlagMap->setAxisTitle("Agreement fraction", 2);

  for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
    m_meSummaryErrorFlagMap->setBinLabel(iSys + 1, m_systemLabel[iSys], 1);
  }
}

void L1EmulatorErrorFlagClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                                      DQMStore::IGetter& igetter,
                                                      const edm::LuminosityBlock& lumiSeg,
                                                      const edm::EventSetup& evSetup) {
  // reset the summary content values
  for (unsigned int iMon = 0; iMon < m_nrL1Systems; ++iMon) {
    m_summaryContent[iMon] = 0.;
  }

  // for masked systems and objects, set the summary content to -1

  for (unsigned int iMon = 0; iMon < m_nrL1Systems; ++iMon) {
    if (m_systemMask[iMon] != 0) {
      m_summaryContent[iMon] = -1;
    }
  }

  // then fill content for unmasked systems

  for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
    float percAgree = -1.;

    if (m_systemMask[iSys] == 0) {
      percAgree = setSummary(igetter, iSys);

      if ((percAgree == -1) && m_verbose) {
        std::cout << "\nWarning: ErrorFlag histogram for system " << m_systemLabel[iSys] << " empty!" << std::endl;
      }
    }

    m_summaryContent[iSys] = percAgree;
  }

  int numUnMaskedSystems = 0;
  for (unsigned int iMon = 0; iMon < m_nrL1Systems; iMon++) {
    if (m_summaryContent[iMon] != -1) {
      numUnMaskedSystems++;
    }
  }

  // fill the SummaryErrorFlagMap histogram for L1 systems
  // (bin 0 - underflow, bin iSys + 1 overflow)
  for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
    m_meSummaryErrorFlagMap->setBinContent(iSys + 1, m_summaryContent[iSys]);
  }

  if (m_verbose) {
    std::cout << "\nSummary report L1EmulatorErrorFlagClient" << std::endl;

    std::cout << "\nL1 systems: " << m_nrL1Systems << " systems included\n" << std::endl;

    for (unsigned int iSys = 0; iSys < m_nrL1Systems; ++iSys) {
      std::cout << std::setw(10) << m_systemLabel[iSys] << std::setw(10) << m_systemLabelExt[iSys] << " \t"
                << m_systemMask[iSys] << " \t" << std::setw(25) << " m_summaryContent[" << std::setw(2) << iSys
                << "] = " << m_summaryContent[iSys] << std::endl;
    }
  }
}

// set subsystem agreement value in summary map
Float_t L1EmulatorErrorFlagClient::setSummary(DQMStore::IGetter& igetter, const unsigned int& iMon) const {
  MonitorElement* QHist = igetter.get(m_systemErrorFlag[iMon]);

  int ntot = 0;
  for (int i = 0; i < QHist->getNbinsX(); i++) {
    ntot += QHist->getBinContent(i + 1);
  }

  bool isEmpty = (ntot == 0);

  //errflag bins: agree, loc agree, loc disagree, data only, emul only

  return isEmpty ? -1. : ((QHist->getBinContent(1)) / (ntot));
}
