#ifndef DQM_L1TMONITORCLIENT_L1TCSCTPGCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TCSCTPGCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class L1TCSCTPGClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TCSCTPGClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TCSCTPGClient() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  void processHistograms(DQMStore::IGetter &);

  std::string monitorDir_;
};

#endif
