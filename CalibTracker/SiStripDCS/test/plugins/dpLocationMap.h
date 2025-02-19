#ifndef CalibTracker_SiStripDCS_test_dplocationmap_H
#define CalibTracker_SiStripDCS_test_dplocationmap_H

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"
#include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMap.h"

class dpLocationMap : public edm::EDAnalyzer {
 public:
  dpLocationMap(const edm::ParameterSet&);
  ~dpLocationMap();
  
 private:
  void beginRun(const edm::Run&, const edm::EventSetup& );
  void analyze( const edm::Event&, const edm::EventSetup& ) {;}

  std::string onlineDbConnectionString;
  std::string authenticationPath;
  coral::TimeStamp tmax, tmin;
  std::vector<int> tmax_par, tmin_par, tDefault;
};
#endif

