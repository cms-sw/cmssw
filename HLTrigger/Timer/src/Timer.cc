// -*- C++ -*-
//
// Package:    Timer
// Class:      Timer
// 
/**\class Timer Timer.cc HLTrigger/Timer/src/Timer.cc

 Description: EDProducer that uses the EventTime structure to store in the Event 
 the names and processing times (per event) for all modules.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Christos Leonidopoulos
//         Created:  Mon Jul 10 14:13:58 CEST 2006
// $Id: Timer.cc,v 1.9 2007/03/27 19:10:58 cleonido Exp $
//
//


#include "HLTrigger/Timer/interface/Timer.h"

#include <iostream>

using edm::ParameterSet; 
using edm::Service;
using edm::ModuleDescription;
//
using edm::ModuleTime;
using edm::EventTime;

using std::string;

Timer::Timer(const ParameterSet& iConfig)
{
  //  produces<EventTime>("Timing");
  produces<EventTime>();
  
  // whether to include timing info about Timer module (default: false)
  includeSelf = iConfig.getUntrackedParameter<bool>("includeSelf", false);

  timing.reset();

  // attach method to Timing service's "new measurement" signal
  Service<TimerService> time;
  time->newMeasurementSignal.connect(boost::bind(boost::mem_fn(&Timer::newTimingMeasurement), this, _1, _2) );
  
  self_module_name = string(iConfig.getParameter<string>("@module_type"));
}

Timer::~Timer()
{
  using namespace std;

  if(!includeSelf){
    string longLine("=========================================================="); 
    cout << longLine << endl;
    cout << " Timer Info:\n";
    cout << " Timer module was excluded from time measurements\n";
    cout << " (to include, set 'bool includeSelf = true' in .cfg file)\n";
    cout << longLine << endl << endl;
  }

}

// fwk calls this method when new module measurement arrives
void Timer::newTimingMeasurement(const ModuleDescription& iMod, double iTime) 
{
  // check if module name corresponds to "this" and skip if needed
  if(!includeSelf && iMod.moduleName() == self_module_name)
     return;

  // new measurement; add to private member
  ModuleTime newModuleTime(iMod.moduleLabel(), iTime); 
  timing.addModuleTime(newModuleTime);
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
Timer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::auto_ptr<EventTime> out(new EventTime(timing));
  // reset data so that we can start from scratch for next event
   timing.reset();
   //
   iEvent.put(out);
}

