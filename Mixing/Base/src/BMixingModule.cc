// File: BMixingModule.cc
// Description:  see BMixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
//
//--------------------------------------------

#include "Mixing/Base/interface/BMixingModule.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ModuleDescription.h"

using namespace std;

int edm::BMixingModule::trackoffset = 0;
int edm::BMixingModule::vertexoffset = 0;

namespace
{
  boost::shared_ptr<edm::PileUp>
  maybeMakePileUp(edm::ParameterSet const& ps)
  {
    boost::shared_ptr<edm::PileUp> pileup; // value to be returned
    // Make sure we have a parameter named 'input'.
    vector<string> names = ps.getParameterNames();
    if (find(names.begin(), names.end(), std::string("input"))
	!= names.end())
      {
	// We have the parameter... make the PileUp
// 	edm::PileUp* pu = new edm::PileUp(ps.getParameter<edm::ParameterSet>("input"));
// 	boost::shared_ptr<edm::PileUp> newguy(pu);
// 	pileup = newguy;
	pileup.reset(new edm::PileUp(ps.getParameter<edm::ParameterSet>("input")));
      }

    return pileup;
  }
}

namespace edm {

  // Constructor 
  BMixingModule::BMixingModule(const edm::ParameterSet& pset) :
    bunchSpace_(pset.getParameter<int>("bunchspace")),
    checktof_(pset.getUntrackedParameter<bool>("checktof",true)),
    input_(maybeMakePileUp(pset)),
    md_()
  {
    md_.pid = pset.id();
    md_.moduleName_ = pset.getParameter<std::string>("@module_type");
    md_.moduleLabel_ = pset.getParameter<std::string>("@module_label");
    //#warning process name is hard coded, for now.  Fix this.
    md_.processName_ = "PILEUP";
    //#warning version and pass are hardcoded
    md_.versionNumber_ = 1;
    md_.pass = 1;
  }

  // Virtual destructor needed.
  BMixingModule::~BMixingModule() { }  

  // Functions that get called by framework every event
  void BMixingModule::produce(edm::Event& e, const edm::EventSetup&) { 

    // Create EDProduct
    createnewEDProduct();

    // Add signals 
    addSignals(e);

    // Read the PileUp
    std::vector<EventPrincipalVector> pileup;
    if ( input_ )
      {
	input_->readPileUp(pileup);
      }

    // Do the merging
    if ( input_ )
      {
	if (input_->doPileup()) LogInfo("PileUp") <<"Adding pileup for event "<<e.id();
	int bunchCrossing = input_->minBunch();
	for (std::vector<EventPrincipalVector>::const_iterator it = pileup.begin();
	     it != pileup.end(); ++it, ++bunchCrossing) {
	  merge(bunchCrossing, *it);
	}
      }

    // Put output into event
    put(e);

  }

  void BMixingModule::merge(const int bcr, const EventPrincipalVector& vec) {
    //
    // main loop: loop over events and merge 
    //
    LogDebug("merge") <<"For bunchcrossing "<<bcr<<", "<<vec.size()<< " events will be merged";
    trackoffset=0;
    vertexoffset=0;
    for (EventPrincipalVector::const_iterator it = vec.begin(); it != vec.end(); ++it) {
      Event e(**it, md_);
      LogDebug("merge") <<" merging Event:  id " << e.id();
      addPileups(bcr, &e);
    }// end main loop
  }

} //edm
