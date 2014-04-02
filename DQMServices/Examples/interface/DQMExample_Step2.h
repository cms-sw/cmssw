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
  void manipulateHistograms(DQMStore::IBooker &, DQMStore::IGetter &) override;  //operation performed in the endJob
  void dqmEndJob();

private:

  //private variables

  //variables from config file
  std::string numMonitorName_;
  std::string denMonitorName_;

  // Histograms
  MonitorElement* h_ptRatio;

};


#endif
