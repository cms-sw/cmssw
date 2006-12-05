// -*- C++ -*-
//
// Package:    HltAnalyzer
// Class:      HltAnalyzer
// 
/**\class HltAnalyzer HltAnalyzer.cc DQM/HLTEvF/src/HltAnalyzer.cc

   Description: Correlate timings and pass/fail for paths and modules 
                on paths.

   Implementation:
     Produces a HLTPerformanceInfo object
*/
//
// Original Author:  Peter Wittich
//         Created:  Thu Nov  9 07:51:28 CST 2006
// $Id: HltAnalyzer.cc,v 1.3 2006/12/04 14:43:38 wittich Exp $
//
//

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/HLTEvF/interface/HltAnalyzer.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HltAnalyzer::HltAnalyzer(const edm::ParameterSet& iConfig)
  : perfInfo_(),
    myName_(iConfig.getParameter<std::string>("@module_label")),
    verbose_(iConfig.getUntrackedParameter("verbose",false)),
    skipNames_(iConfig.getParameter<std::vector<std::string> >("skipNames")),
    trigResLabel_(iConfig.getParameter<edm::InputTag>("triggerResultsLabel"))
{
  // now do what ever initialization is needed
  produces<HLTPerformanceInfo>();

  // this object needs to exist outside of the scope of the event 
  // entry point so that the EDM can call it when modules run
  perfInfo_.clear();
  
  // attach method to Timing service's "new measurement" signal
  edm::Service<edm::service::Timing> time;
  time->newMeasurementSignal.
    connect(boost::bind(boost::mem_fn(&HltAnalyzer::newTimingMeasurement), 
			this, _1, _2) );


}


HltAnalyzer::~HltAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HltAnalyzer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Before we get here, the timing service call-back will have filled the 
  // module information in the perfInfo_ member variable.
  // Beware, though, that this can fail if the module in question did not run. 

  using namespace edm;
  Handle<TriggerResults> pTrig;
  try {
    iEvent.getByLabel(trigResLabel_, pTrig );
  } 
  catch(...) { // and drop...
    ;
  }
  if (pTrig.isValid()) {
    //LogDebug(myName_) << "TriggerResults " << (*i)
    if ( verbose() ) 
      std::cout << "TriggerResults " << trigResLabel_
		<< " found, number of HLT paths: " 
		<< pTrig->size() << std::endl;;
  } else {
    //LogDebug(myName_) << "TriggerResults " << (*i) << " product not found - "
    std::cout<< "TriggerResults product not found - "<< trigResLabel_
	     << "returning result=false!" << std::endl;
    // clear for next event
    perfInfo_.clear();
    return false;
  }

  // assign modules to paths.  The perfInfo module might already contain
  // an entry for this module with timing. in that case, the module is just
  // added to the paths. If it does not, a module with time = 0 is added.
  using edm::service::TriggerNamesService;
  Service<TriggerNamesService> trigger_paths;
  TriggerNamesService::Strings paths = trigger_paths->getTrigPaths();
  if ( verbose() ) {
    std::cout << "Dumping paths." << std::endl;
  }
  // Loop over paths
  for ( TriggerNamesService::Strings::const_iterator i = paths.begin();
	i != paths.end(); ++i ) {
    HLTPerformanceInfo::Path p(*i);
    // this sets the status for the path
    unsigned int where = pTrig->find(*i);
    p.setStatus( pTrig->at(where));
    if ( verbose() ) {
      std::cout << "Path is " << *i
		<< " with result " << p.status().state()
		<< std::endl;
    }
    // get a list of the names of the modules on the path in question
    TriggerNamesService::Strings 
      mods_on_path = trigger_paths->getTrigPathModules(*i);
    // loop over modules on the path
    for ( TriggerNamesService::Strings::const_iterator 
	    j = mods_on_path.begin();
	  j != mods_on_path.end(); ++j ) {
      if ( verbose() ) {
	std::cout << "\tmodule is " << *j << std::endl;
      }
      perfInfo_.addModuleToPath(j->c_str(), &p); 
    }
    perfInfo_.addPath(p);
  }

  // DEBUG ONLY: PRINT OUT ALL MODULES
  if ( verbose() ) {
    std::cout << myName_<< ": dumping modules internal to perfinfo: " 
	      << perfInfo_.numberOfModules()
	      << std::endl;
    for ( HLTPerformanceInfo::Modules::const_iterator i = 
	    perfInfo_.beginModules(); 
	  i != perfInfo_.endModules(); 
	  ++i ) {
      std::cout << i->name() << ": " << i->time() 
		<< ":  " << i->status().state()
		<< std::endl;
    }
    std::cout << myName_ << ": dumping path times.... " << std::endl;
    for ( HLTPerformanceInfo::PathList::const_iterator j = 
	    perfInfo_.beginPaths(); 
	  j != perfInfo_.endPaths(); ++j ) {
      std::cout << "\t" << j->name() << ": " 
		<< j->time()  << " " 
		<< j->status().state() << " "
		<< j->lastModuleByStatusName() << "\t"
		<< j->lastModuleByStatus()->time() 
		<< std::endl;
    }
  }
  // END DEBUG

  // 

 
  // done - now store
  std::auto_ptr<HLTPerformanceInfo> pPerf(new HLTPerformanceInfo(perfInfo_));
   
  iEvent.put(pPerf);

  // clear for next event
  perfInfo_.clear();

  return true;
}

// ------- method called once each job just before starting event loop  -----
void 
HltAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------ method called once each job just after ending the event loop  ----
void 
HltAnalyzer::endJob() {
}


// fwk calls this method when new module measurement arrives
void HltAnalyzer::newTimingMeasurement(const edm::ModuleDescription& iMod, 
				       double diffTime) 
{
  // skip this module if it's in the skip list
  if ( std::find(skipNames_.begin(), skipNames_.end(), iMod.moduleLabel())
       != skipNames_.end() ) 
    return;
  //
  HLTPerformanceInfo::Module m(iMod.moduleLabel().c_str(), diffTime);
  perfInfo_.addModule(m);
  if ( verbose() ) {
    std::cout << myName_ << ": adding module with name " << iMod.moduleLabel()
	      << " and time " << diffTime 
	      << ", size " << perfInfo_.numberOfModules()
	      << std::endl;
  }


}



