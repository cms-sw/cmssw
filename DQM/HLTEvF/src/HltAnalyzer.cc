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
// $Id$
//
//



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
    verbose_(iConfig.getUntrackedParameter("verbose",false))
{
  // now do what ever initialization is needed
  produces<HLTPerformanceInfo>();

  // this object needs to exist outside of the scope of the event 
  // entry point so that the EDM can call it when modules run
  perfInfo_.clear();
  
  trigResLabel_ = iConfig.getParameter< edm::InputTag > ("triggerResultsLabel");

  // attach method to Timing service's "new measurement" signal
  edm::Service<edm::service::Timing> time;
  time->newMeasurementSignal.connect(boost::bind(boost::mem_fn(&HltAnalyzer::newTimingMeasurement), this, _1, _2) );


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
  iEvent.getByType(pTrig);
  //iEvent.getByLabel(trigResLabel_,pTrig);


  using edm::service::TriggerNamesService;
  Service<TriggerNamesService> trigger_paths;
  TriggerNamesService::Strings paths = trigger_paths->getTrigPaths();
  if ( verbose() ) {
    std::cout << "Dumping paths." << std::endl;
  }
  for ( TriggerNamesService::Strings::const_iterator i = paths.begin();
	i != paths.end(); ++i ) {
    HLTPerformanceInfo::Path p(*i);
    unsigned int where = pTrig->find(*i);
    p.setStatus( pTrig->at(where));
    if ( verbose() ) {
      std::cout << "Path is " << *i
		<< " with result " << p.Status().state()
		<< std::endl;
    }
    TriggerNamesService::Strings 
      mods_on_path = trigger_paths->getTrigPathModules(*i);
    for ( TriggerNamesService::Strings::const_iterator 
	    j = mods_on_path.begin();
	  j != mods_on_path.end(); ++j ) {
      if ( verbose() ) 
	std::cout << "module is " << *j << std::endl;
      // this call could fail if the module didn't run this event.
      perfInfo_.addModuleToPath(j->c_str(), &p); 
    }
    perfInfo_.addPath(p);
  }

  if ( verbose() ) {
    const HLTPerformanceInfo::Modules *l = perfInfo_.ListOfModules();
    std::cout << myName_<< ": dumping modules internal to perfinfo: " 
	      << l->size()
	      << std::endl;
    for ( HLTPerformanceInfo::Modules::const_iterator i = l->begin(); 
	  i != l->end(); ++i ) {
      std::cout << i->Name() << ": " << i->Time() << std::endl;
    }
    std::cout << myName_ << ": dumping path times.... " << std::endl;
    const HLTPerformanceInfo::PathList *p = perfInfo_.ListOfPaths();
    for ( HLTPerformanceInfo::PathList::const_iterator j = p->begin();
	  j != p->end(); ++j ) {
      std::cout << "\t" << j->Name() << ": " << j->Time() << std::endl;
    }
  }

 
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
  HLTPerformanceInfo::Module m(iMod.moduleLabel().c_str(), diffTime);
  perfInfo_.addModule(m);
  if ( verbose() ) {
    std::cout << myName_ << ": adding module with name " << iMod.moduleLabel()
	      << " and time " << diffTime 
	      << ", size " << perfInfo_.ListOfModules()->size()
	      << std::endl;
  }


}



