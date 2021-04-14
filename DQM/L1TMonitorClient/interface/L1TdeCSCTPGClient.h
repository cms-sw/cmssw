#ifndef DQM_L1TMONITORCLIENT_L1TdeCSCTPGCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TdeCSCTPGCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class L1TdeCSCTPGClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TdeCSCTPGClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TdeCSCTPGClient() override;

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

  std::vector<std::string> alctVars_;
  std::vector<std::string> clctVars_;
  std::vector<std::string> lctVars_;

  std::vector<unsigned> alctNBin_;
  std::vector<unsigned> clctNBin_;
  std::vector<unsigned> lctNBin_;
  std::vector<double> alctMinBin_;
  std::vector<double> clctMinBin_;
  std::vector<double> lctMinBin_;
  std::vector<double> alctMaxBin_;
  std::vector<double> clctMaxBin_;
  std::vector<double> lctMaxBin_;

  // first key is the chamber number
  // second key is the variable
  std::map<uint32_t, std::map<std::string, MonitorElement *> > chamberHistos_;
};

#endif
