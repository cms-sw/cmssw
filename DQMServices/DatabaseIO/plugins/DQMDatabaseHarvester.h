#ifndef DQMExample_Step2DB_h
#define DQMExample_Step2DB_h

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//DQM
#include "DQMServices/Core/interface/DQMDbHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class DQMExample_Step2DB: public DQMDbHarvester{

public:

  DQMExample_Step2DB(const edm::ParameterSet& ps);
  virtual ~DQMExample_Step2DB();

protected:

  void beginJob();
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&);  //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);


private:

  //private variables

  //variables from config file
  std::string numMonitorName_;
  std::string denMonitorName_;

  std::string s_histogramsPath;
  std::vector <std::string> vs_histogramsPerLumi;
  std::vector <std::string> vs_histogramsPerRun;
  std::vector <MonitorElement *> histogramsPerLumi;
  std::vector < std::pair <MonitorElement *, valuesOfHistogram> > histogramsPerRun;
  // Histograms
  MonitorElement* h_ptRatio;
/**/
};


#endif
