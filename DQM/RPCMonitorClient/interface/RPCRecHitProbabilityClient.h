#ifndef RPCRecHitProbabilityClient_H
#define RPCRecHitProbabilityClient_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include <FWCore/Framework/interface/ESHandle.h>


class RPCRecHitProbabilityClient:public  DQMEDHarvester{

public:

  /// Constructor
  RPCRecHitProbabilityClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~ RPCRecHitProbabilityClient();
  
  
protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

 


 private:

    std::string  globalFolder_;
  
 
  
};
#endif
