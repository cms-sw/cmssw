#ifdef __linux__
#include "PerfTools/PerfStat/interface/PerfStat.h" 
#else
#include<ostream>
struct PerfStat {
  PerfStat(){}
  void start(){}
  void stop(){}
  void summary(std::ostream &) const{}
  static void header(std::ostream &){}
};
#endif

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"


#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <iostream>
#include <sstream>
#include <unordered_map>


class PerfStatService {

public:

  PerfStatService(edm::ParameterSet const&,edm::ActivityRegistry&);
  
private:

  void postBeginJob();
  void postEndJob();
  
  void preEventProcessing(edm::EventID const&, edm::Timestamp const&){}
  void postEventProcessing(edm::Event const&, edm::EventSetup const&){ master.start();master.stop();}
  
  void preModule(edm::ModuleDescription const& md) {
    current = &find(md); current->startDelta();
  }
  void postModule(edm::ModuleDescription const& md) {
    current->stopDelta();
  }

  // find or create
  PerfStat & find(edm::ModuleDescription const& md) {
    auto p = perfs.find(md.parameterSetID().compactForm());
    if (p==perfs.end()) {
      std::string name = md.moduleName()+'/'+md.moduleLabel();
      p = perfs.insert(std::make_pair(md.parameterSetID().compactForm(),P(name,master))).first;
    }
    return (*p).second.p;
  }

  

private:
  PerfStat master;
  struct P {
    P(std::string in, PerfStat const & im) : name(in), p(im.fd()){}
    std::string name;
    PerfStat p;
  };
  std::unordered_map<std::string, P> perfs;
  PerfStat * current = nullptr;

};


PerfStatService::PerfStatService(edm::ParameterSet const& iPS, edm::ActivityRegistry& iRegistry) {
  
  iRegistry.watchPostBeginJob(this, &PerfStatService::postBeginJob);
  iRegistry.watchPostEndJob(this, &PerfStatService::postEndJob);
  
  iRegistry.watchPreProcessEvent(this, &PerfStatService::preEventProcessing);
  iRegistry.watchPostProcessEvent(this, &PerfStatService::postEventProcessing);
  
  iRegistry.watchPreModule(this, &PerfStatService::preModule);
  iRegistry.watchPostModule(this, &PerfStatService::postModule);
  
  
}



void PerfStatService::postBeginJob() {
  
}

void PerfStatService::postEndJob() {
  std::ostringstream out;
  out.precision(3);
  out.setf( std::ios::fixed, std:: ios::floatfield );
  out << "|module  ";
  PerfStat::header(out);
  master.read(); master.calib();
  out << "|Total  "; master.summary(out,1.e-6,100.);
  for ( auto const & p : perfs) {
    out << '|' << p.second.name << "  "; p.second.p.summary(out,1.e-6,100.);
  }
  std::cout << out.str() << std::endl;

}

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(PerfStatService);
