// File: BMixingModule.cc
// Description:  see BMixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "Mixing/Base/interface/BMixingModule.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/src/SecondaryInputSourceFactory.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "Mixing/Base/interface/FixedPUGenerator.h"
#include "Mixing/Base/interface/PoissonPUGenerator.h"

using namespace std;

int edm::BMixingModule::trackoffset = 0;
int edm::BMixingModule::vertexoffset = 0;

namespace edm
{

  // Constructor 
  BMixingModule::BMixingModule(const edm::ParameterSet& ps) : 
     average_(0.), minbunch_(ps.getParameter<int>("minbunch")), maxbunch_(ps.getParameter<int>("maxbunch"))
   
  {
    bunchSpace_=ps.getParameter<int>("bunchspace");


    // test for factory =============================
    //    std::string type =ps.getParameter<string>("type");
    //    double average = ps.getParameter<double>("average_number");
    //    std::auto_ptr<PUGenerator> pugen_
    //      (PUFactory::get()->makePUGenerator(ps).release());
    // end of test for factory =============================

    if (!strcmp(ps.getParameter<string>("type").c_str(),"fixed"))
      generator_ = new FixedPUGenerator(int(ps.getParameter<double>
					    ("average_number")));
    else
      generator_ = new PoissonPUGenerator(ps.getParameter<double>("average_number"));

    std::cout << "Constructed a mixing module with: average = "<<average_<<" ,minbunch = "<<minbunch_<<" ,maxbunch = "<<maxbunch_<<std::endl;
 
    // make secondary input source
    secInput_ =makeSecInput(ps);
  }

  // Virtual destructor needed.
  BMixingModule::~BMixingModule() { }  

  
  // Functions that get called by framework every event
  void BMixingModule::produce(edm::Event& e, const edm::EventSetup&)
  { 

    cout <<"\n==============================>  Start produce for event "<<e.id()<<endl;
    // Create EDProduct
    createnewEDProduct();

    // Add signals 
    addSignals(e);

    // Do the merging
    for (int bunchCrossing=minbunch_;bunchCrossing<=maxbunch_;++bunchCrossing) {
      getEvents(generator_->numberOfEventsPerBunch());
      merge(bunchCrossing, eventVector_);
      eventVector_.clear();
    }

    // Put output into event
    put(e);

  }


  boost::shared_ptr<SecondaryInputSource> BMixingModule::makeSecInput(ParameterSet const& ps)
  {
    // This is a temporary version, waiting for the random input service to be written
    ParameterSet sec_input = ps.getParameter<ParameterSet>("input");

    boost::shared_ptr<SecondaryInputSource> input_(static_cast<SecondaryInputSource *>
						   (SecondaryInputSourceFactory::get()->makeSecondaryInputSource(sec_input).release()));
    return input_;
  }

  void BMixingModule::getEvents (const unsigned int nrEvents)
  {
    // filling of eventvector by using  secondary input source 
    unsigned int eventCount=0;
    //    eventVector_.clear();

    std::vector<EventPrincipal*> result;
    secInput_->readMany(0, nrEvents, result);
    while (eventCount < nrEvents) {
      ModuleDescription md=ModuleDescription();  //temporary
      Event *event = new Event(*result[eventCount], md);
      //      cout <<"\n Pileup event nr "<<eventCount<<" event id "<<event->id()<<endl;
      eventVector_.push_back (event);
      ++eventCount;
    }
    cout <<endl;
  }

  void BMixingModule::merge(const int bcr, const std::vector<Event *> vec){
    //
    // main loop: loop over events and merge 
    //
    cout <<endl<<" For bunchcrossing "<<bcr<<",  "<<vec.size()<< " events will be merged"<<flush<<endl;
    trackoffset=0;
    vertexoffset=0;
    for (std::vector<Event *>::const_iterator it =vec.begin(); it != vec.end(); it++)    {
      cout <<" merging Event:  id "<<(*it)->id()<<flush<<endl;
      addPileups(bcr,(*it));

      // delete the event
      delete (*it);

    }// end main loop
  }
  

} //edm
