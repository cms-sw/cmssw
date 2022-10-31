#ifndef DQM_L1TMONITORCLIENT_L1TdeGEMTPGCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TdeGEMTPGCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class L1TdeGEMTPGClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TdeGEMTPGClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TdeGEMTPGClient() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  void book(DQMStore::IBooker &ibooker);
  void processHistograms(DQMStore::IGetter &);

  std::string monitorDir_;

  std::vector<std::string> chambers_;

  std::vector<std::string> clusterVars_;
  std::vector<unsigned> clusterNBin_;
  std::vector<double> clusterMinBin_;
  std::vector<double> clusterMaxBin_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement *> > chamberHistos_;
};

#endif
