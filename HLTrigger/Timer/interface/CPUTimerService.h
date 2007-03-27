#ifndef CPUTimer_Service_
#define CPUTimer_Service_

/**\class CPUTimerService

Description: Class accessing CPUTimer to record CPU-processing-time info per module

Original Author:  Christos Leonidopoulos, March 2007

*/

#include "FWCore/Utilities/interface/CPUTimer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <string>

class CPUTimerService {
 public:
  explicit CPUTimerService(const edm::ParameterSet&,
			   edm::ActivityRegistry& iAR);
  ~CPUTimerService();
  // fwk calls this method before a module is processed
  void preModule(const edm::ModuleDescription& iMod);
  // fwk calls this method after a module has been processed
  void postModule(const edm::ModuleDescription& iMod);
  // get cpu-processing time for last module that ran (in secs)
  double getTime() const{return cpu_time;}
  // get name of last module that ran
  std::string getModuleName() const{return module_name;}

 private:
  // cpu-timer
  static edm::CPUTimer * cpu_timer;
  // module name for last measurement
  std::string module_name;
  // cpu-processing time for last module (secs)
  double cpu_time;

};

#endif // #define CPUTimer_Service_
