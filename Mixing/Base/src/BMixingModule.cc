// File: BMixingModule.cc
// Description:  see BMixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
//
//--------------------------------------------

#include "Mixing/Base/interface/BMixingModule.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

using namespace std;

int edm::BMixingModule::trackoffset = 0;
int edm::BMixingModule::vertexoffset = 0;

namespace edm {

  // Constructor 
  BMixingModule::BMixingModule(const edm::ParameterSet& pset) :
    bunchSpace_(pset.getParameter<int>("bunchspace")),
    checktof_(pset.getUntrackedParameter<bool>("checktof",true)),
    input_(pset.getParameter<ParameterSet>("input")),
    md_()
  {
    md_.pid = pset.id();
    md_.moduleName_ = pset.getUntrackedParameter<std::string>("@module_type");
    md_.moduleLabel_ = pset.getUntrackedParameter<std::string>("@module_label");
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

    //    cout <<"\n==============================>  Start produce for event " << e.id() << endl;
    // Create EDProduct
    createnewEDProduct();

    // Add signals 
    addSignals(e);

    // Read the PileUp
    std::vector<EventPrincipalVector> pileup;
    input_.readPileUp(pileup);

    // Do the merging
    int bunchCrossing = input_.minBunch();
    for (std::vector<EventPrincipalVector>::const_iterator it = pileup.begin();
        it != pileup.end(); ++it, ++bunchCrossing) {
      merge(bunchCrossing, *it);
    }

    // Put output into event
    put(e);

  }

  void BMixingModule::merge(const int bcr, const EventPrincipalVector& vec) {
    //
    // main loop: loop over events and merge 
    //
    //    cout <<endl<<" For bunchcrossing "<<bcr<<",  "<<vec.size()<< " events will be merged"<<flush<<endl;
    trackoffset=0;
    vertexoffset=0;
    for (EventPrincipalVector::const_iterator it = vec.begin(); it != vec.end(); ++it) {
      Event e(**it, md_);
      //      cout <<" merging Event:  id " << e.id() << flush << endl;
      addPileups(bcr, &e);
    }// end main loop
  }

} //edm
