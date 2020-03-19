/*!
  \file RPCClient.h
   \author A. Cimmino
*/
#ifndef RPCClient_H
#define RPCClient_H

#include "DQMServices/Core/interface/DQMStore.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <FWCore/Framework/interface/Event.h>
//#include <FWCore/Framework/interface/Run.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include <FWCore/Framework/interface/LuminosityBlock.h>
//#include "FWCore/ServiceRegistry/interface/Service.h"

//#include <map>
#include <vector>
#include <string>

class RPCClient {
public:
  typedef dqm::harvesting::DQMStore DQMStore;
  typedef dqm::harvesting::MonitorElement MonitorElement;

  //RPCClient(const edm::ParameterSet& ps) {}
  virtual ~RPCClient(void) {}

  virtual void clientOperation() = 0;

  virtual void getMonitorElements(std::vector<MonitorElement *> &, std::vector<RPCDetId> &, std::string &) = 0;

  virtual void beginJob(std::string &) = 0;

  virtual void myBooker(DQMStore::IBooker &) = 0;
};

#endif
