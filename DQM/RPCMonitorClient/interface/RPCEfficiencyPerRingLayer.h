#ifndef RPCEfficiencyPerRingLayer_H
#define RPCEfficiencyPerRingLayer_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include <string>

class RPCEfficiencyPerRingLayer:public DQMEDHarvester{
public:

  /// Constructor
  RPCEfficiencyPerRingLayer(const edm::ParameterSet& iConfig);
  
  /// Destructor
  virtual ~RPCEfficiencyPerRingLayer();

  
 protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob


  
 private:

  MonitorElement * EfficiencyPerRing;
  MonitorElement * EfficiencyPerLayer;  

  int  numberOfDisks_;
  int innermostRings_ ;

  std::string globalFolder_;

};

#endif
