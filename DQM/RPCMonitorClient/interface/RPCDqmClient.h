#ifndef RPCDqmClient_H
#define RPCDqmClient_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/RPCMonitorClient/interface/RPCClient.h"

#include <string>
#include <vector>

class RPCDqmClient:public  DQMEDHarvester { 

 public:

  /// Constructor
 RPCDqmClient(const edm::ParameterSet& ps);
  
  /// Destructor
  ~RPCDqmClient() override;

 protected:

 void beginJob() override;
 void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; //performed in the endLumi
 void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

  void makeClientMap(const edm::ParameterSet& parameters_);
  void getMonitorElements( DQMStore::IGetter & );
  void getRPCdetId( const edm::EventSetup& );

 private:

  bool offlineDQM_;
  int prescaleGlobalFactor_, minimumEvents_, numLumBlock_;
 
  bool useRollInfo_,enableDQMClients_ ; 
  std::string  prefixDir_;
  std::string  globalFolder_;
  std::vector<std::string>  clientList_;
  int lumiCounter_;
  MonitorElement * RPCEvents_; 
  std::vector<RPCDetId>  myDetIds_;
  std::vector<std::string> clientNames_,clientHisto_; 
  std::vector<RPCClient*> clientModules_;

  std::vector<int> clientTag_;
    
};
#endif
