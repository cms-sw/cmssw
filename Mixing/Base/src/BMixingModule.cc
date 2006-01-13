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
      input_(pset.getParameter<ParameterSet>("input")) {
    bunchSpace_ = pset.getParameter<int>("bunchspace");
  }

  // Virtual destructor needed.
  BMixingModule::~BMixingModule() { }  

  // Functions that get called by framework every event
  void BMixingModule::produce(edm::Event& e, const edm::EventSetup&) { 

    cout <<"\n==============================>  Start produce for event " << e.id() << endl;
    // Create EDProduct
    createnewEDProduct();

    // Add signals 
    addSignals(e);

    // Read the PileUp
    typedef std::vector<std::vector<Event *> > Events;
    Events pileup;
    input_.readPileUp(pileup);

    // Do the merging
    int bunchCrossing = input_.minBunch();
    for (Events::const_iterator it = pileup.begin(); it != pileup.end(); ++it, ++bunchCrossing) {
      merge(bunchCrossing, *it);
    }

    // Put output into event
    put(e);

  }

  void BMixingModule::merge(const int bcr, const std::vector<Event *> vec) {
    //
    // main loop: loop over events and merge 
    //
    //    cout <<endl<<" For bunchcrossing "<<bcr<<",  "<<vec.size()<< " events will be merged"<<flush<<endl;
    trackoffset=0;
    vertexoffset=0;
    for (std::vector<Event *>::const_iterator it =vec.begin(); it != vec.end(); it++) {
      //      cout <<" merging Event:  id "<<(*it)->id()<<flush<<endl;
      addPileups(bcr,(*it));

      // delete the event
      delete (*it);

    }// end main loop
  }

} //edm
