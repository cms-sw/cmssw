#ifndef DQM_L1TMONITORCLIENT_L1TRPCTFClient_H
#define DQM_L1TMONITORCLIENT_L1TRPCTFClient_H

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
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile2D.h>

class L1TRPCTFClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TRPCTFClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TRPCTFClient() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  void initialize();
  void book(DQMStore::IBooker &ibooker);
  void processHistograms(DQMStore::IGetter &igetter);

  MonitorElement *m_phipackedbad;
  MonitorElement *m_phipackeddead;
  MonitorElement *m_deadChannels;
  MonitorElement *m_noisyChannels;

  edm::ParameterSet parameters_;
  std::string monitorName_;
  std::string input_dir_;
  std::string output_dir_;
  int counterLS_;    ///counter
  int counterEvt_;   ///counter
  int prescaleLS_;   ///units of lumi sections
  int prescaleEvt_;  ///prescale on number of events

  bool verbose_;

  bool m_runInEventLoop;
  bool m_runInEndLumi;
  bool m_runInEndRun;
  bool m_runInEndJob;
};

#endif
