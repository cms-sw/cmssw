#ifndef RunInfoAdder_H
#define RunInfoAdder_H

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
 
class RunInfoAdder : public DQMEDHarvester{

public:

  RunInfoAdder(const edm::ParameterSet& ps);
  virtual ~RunInfoAdder();
  
protected:

  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&);  //performed in the endLumi
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:

  //variables from config file
  bool addRunNumber_;
  bool addLumi_;
  std::vector<std::string> folder_;


  // information to add, has to be saved from endLumi
  uint32_t run_;

};


#endif
