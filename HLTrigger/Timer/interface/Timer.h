#ifndef HighLevelTrigger_Timer_h
#define HighLevelTrigger_Timer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HLTReco/interface/ModuleTiming.h"
// use for CMSSW_1_3_x
//#include "DataFormats/Common/interface/ModuleDescription.h"
// use for CMSSW_1_4_x
#include "DataFormats/Provenance/interface/ModuleDescription.h"


#include "HLTrigger/Timer/interface/TimerService.h"

/*
  Description: EDProducer that uses the EventTime structure to store in the Event 
  the names and processing times (per event) for all modules.
  
  Implementation:
  <Notes on implementation>
*/
//
// Original Author:  Christos Leonidopoulos
//         Created:  Mon Jul 10 14:13:58 CEST 2006
// $Id: Timer.h,v 1.11 2007/03/27 19:12:13 cleonido Exp $
//
//
//
// class decleration
//

class Timer : public edm::EDProducer {
 public:
  explicit Timer(const edm::ParameterSet&);
  ~Timer();
  // fwk calls this method when new module measurement arrives;
  void newTimingMeasurement(const edm::ModuleDescription& iMod, double iTime);
  // put output into Event
  virtual void produce(edm::Event&, const edm::EventSetup&);
  //
 private:
  // ----------member data ---------------------------
  edm::EventTime timing; // structure with module names & processing times
  // whether to store information about itself (ie. Timer module)
  bool includeSelf;

  // this should be just this class' name
  std::string self_module_name;

};


#endif // #define HighLevelTrigger_Timer_h
