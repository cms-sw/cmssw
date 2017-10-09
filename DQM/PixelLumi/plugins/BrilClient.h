#ifndef DQM_BRIL_BRILCLIENT_H
#define DQM_BRIL_BRILCLIENT_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <map>

class BrilClient : public DQMEDHarvester {
 public:
  BrilClient(const edm::ParameterSet &ps);
  virtual ~BrilClient();

 protected:
  void beginJob() override{};
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override{};

 private:
  edm::EDGetTokenT<std::string> pathToken_;
  edm::EDGetTokenT<std::string> jsonToken_;
};

#endif
