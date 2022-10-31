#ifndef DQM_L1TMONITORCLIENT_L1TCSCTFCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TCSCTFCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class L1TCSCTFClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TCSCTFClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TCSCTFClient() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  void initialize();

  void processHistograms(DQMStore::IGetter &);

  edm::ParameterSet parameters;

  std::string input_dir, output_dir;
  int counterLS;    ///counter
  int counterEvt;   ///counter
  int prescaleLS;   ///units of lumi sections
  int prescaleEvt;  ///prescale on number of events

  bool m_runInEventLoop;
  bool m_runInEndLumi;
  bool m_runInEndRun;
  bool m_runInEndJob;

  // -------- member data --------
  MonitorElement *csctferrors_;
};

#endif
