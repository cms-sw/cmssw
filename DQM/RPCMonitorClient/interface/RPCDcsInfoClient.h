#ifndef RPCDCSINFOCLIENT_H
#define RPCDCSINFOCLIENT_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RPCDcsInfoClient : public DQMEDHarvester {
public:
  RPCDcsInfoClient(const edm::ParameterSet &ps);
  ~RPCDcsInfoClient() override = default;

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  const std::string dcsinfofolder_;
  const std::string eventinfofolder_;
  const std::string dqmprovinfofolder_;

};

#endif
