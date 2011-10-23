#ifndef Timer_Service_
#define Timer_Service_

/**\class TimerService

Description: Class accessing CPUTimer to record processing-time info per module
(either CPU-time or wall-clock-time)

Original Author:  Christos Leonidopoulos, March 2007

*/

#include "sigc++/signal.h"

#include "FWCore/Utilities/interface/CPUTimer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <string>

class TimerService {
 public:
  TimerService(const edm::ParameterSet&, edm::ActivityRegistry& iAR);
  ~TimerService();
  // signal with module-description and processing time (in secs)
  sigc::signal<void, const edm::ModuleDescription&, double> newMeasurementSignal;

  // fwk calls this method before a module is processed
  void preModule(const edm::ModuleDescription& iMod);
  // fwk calls this method after a module has been processed
  void postModule(const edm::ModuleDescription& iMod);

 private:
  // whether to use CPU-time (default) or wall-clock time
  bool useCPUtime;
  // cpu-timer
  edm::CPUTimer cpu_timer; // Chris J's CPUTimer
};

#endif // #define Timer_Service_
