#ifndef RPCRecHitProbabilityClient_H
#define RPCRecHitProbabilityClient_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

class RPCRecHitProbabilityClient : public DQMEDHarvester {
public:
  RPCRecHitProbabilityClient(const edm::ParameterSet &ps);
  ~RPCRecHitProbabilityClient() override = default;

protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  std::string globalFolder_;
};
#endif
