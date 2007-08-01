#ifndef ECALTRENDCHECKER_H
#define ECALTRENDCHECKER_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"


#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;
using namespace oracle::occi;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class  EcalTrendChecker : public edm::EDAnalyzer {
 public:
  EcalCondDBInterface* econn;

  explicit  EcalTrendChecker(const edm::ParameterSet& iConfig );
  ~EcalTrendChecker();


  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);


  

 private:
 
  unsigned long m_firstRun ;
  unsigned long m_lastRun ;

  std::string sid;
  std::string user;
  std::string pass;


};

#endif
