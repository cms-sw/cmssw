#ifndef DQM_L1TMONITORCLIENT_L1EmulatorErrorFlagClient_H
#define DQM_L1TMONITORCLIENT_L1EmulatorErrorFlagClient_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class L1EmulatorErrorFlagClient : public DQMEDHarvester {
public:
  /// Constructor
  L1EmulatorErrorFlagClient(const edm::ParameterSet &);

  /// Destructor
  ~L1EmulatorErrorFlagClient() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  /// input parameters

  bool m_verbose;
  std::vector<edm::ParameterSet> m_l1Systems;
  std::vector<std::string> m_maskL1Systems;

  bool m_runInEventLoop;
  bool m_runInEndLumi;
  bool m_runInEndRun;
  bool m_runInEndJob;

  /// private methods

  void initialize();

  Float_t setSummary(DQMStore::IGetter &igetter, const unsigned int &) const;

  /// number of L1 trigger systems
  size_t m_nrL1Systems;

  std::vector<std::string> m_systemLabel;
  std::vector<std::string> m_systemLabelExt;
  std::vector<int> m_systemMask;
  std::vector<std::string> m_systemFolder;

  std::vector<std::string> m_systemErrorFlag;

  /// summary report

  std::vector<Float_t> m_summaryContent;
  MonitorElement *m_meSummaryErrorFlagMap;
};

#endif
