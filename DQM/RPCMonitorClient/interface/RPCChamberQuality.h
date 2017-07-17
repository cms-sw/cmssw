#ifndef RPCChamberQuality_H
#define RPCChamberQuality_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class RPCChamberQuality:public DQMEDHarvester{
 public:
  
  RPCChamberQuality(const edm::ParameterSet& ps);
  virtual ~RPCChamberQuality();
  
  
 protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

 private:

  void fillMonitorElements( DQMStore::IGetter &);

  void myBooker(DQMStore::IBooker &);

  MonitorElement * RpcEvents;
  enum chamberQualityState { GoodState= 1 , OffState =2, NoisyStripState= 3, NoisyRollState= 4 , PartiallyDeadState=5 , DeadState=6,BadShapeState=7 };

  int lumiCounter_;

  bool  enableDQMClients_;
  bool offlineDQM_;

  void performeClientOperation(std::string , int , MonitorElement *,  DQMStore::IGetter& );
  
  std::string prefixDir_, summaryDir_;
  static const std::string xLabels_[7];
  static const std::string regions_[3];
  bool useRollInfo_;
  int prescaleFactor_;
  int numberOfDisks_;

  bool init_;

  int minEvents;
  int numLumBlock_;
};

#endif
