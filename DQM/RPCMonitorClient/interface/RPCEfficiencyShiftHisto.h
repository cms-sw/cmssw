#ifndef RPCEfficiencyShiftHisto_H
#define RPCEfficiencyShiftHisto_H

// * *
// *  RPCEfficiencyShiftHisto
// * *

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include <string>


class DQMStore;
class RPCDetId;


class RPCEfficiencyShiftHisto:public DQMEDHarvester{
public:

  /// Constructor
  RPCEfficiencyShiftHisto(const edm::ParameterSet& iConfig);
  
  /// Destructor
  virtual ~RPCEfficiencyShiftHisto();

 protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

  
 private:


  MonitorElement * EffBarrelRoll;
  MonitorElement * EffEndcapPlusRoll;
  MonitorElement * EffEndcapMinusRoll;
  MonitorElement * RollPercentage;
  
  int  numberOfDisks_;
  int effCut_;

  std::string globalFolder_;

};

#endif
