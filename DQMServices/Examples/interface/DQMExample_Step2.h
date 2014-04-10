#ifndef DQMExample_Step2_H
#define DQMExample_Step2_H

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
 
class DQMExample_Step2: public DQMEDHarvester{

public:

  DQMExample_Step2(const edm::ParameterSet& ps);
  virtual ~DQMExample_Step2();
  
protected:

  void beginJob();
  void dqmEndLuminosityBlock(DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&);  //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:

  //private variables

  //variables from config file
  std::string numMonitorName_;
  std::string denMonitorName_;

  // Histograms
  MonitorElement* h_ptRatio;

};


#endif
