#ifndef RPCDCSINFOCLIENT_H
#define RPCDCSINFOCLIENT_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RPCDcsInfoClient : public DQMEDHarvester { 

public:
  RPCDcsInfoClient( const edm::ParameterSet& ps);
  ~RPCDcsInfoClient() override;

protected:

 void beginJob() override;
 void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; //performed in the endLumi
 void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob


private:

  std::string dcsinfofolder_;

   std::vector<int> DCS;
};

#endif
