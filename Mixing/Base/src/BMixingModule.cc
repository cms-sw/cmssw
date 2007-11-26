// File: BMixingModule.cc
// Description:  see BMixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
//
//--------------------------------------------

#include "Mixing/Base/interface/BMixingModule.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

using namespace std;

int edm::BMixingModule::vertexoffset = 0;
const unsigned int edm::BMixingModule::maxNbSources =3;

namespace
{
  boost::shared_ptr<edm::PileUp>
  maybeMakePileUp(edm::ParameterSet const& ps,std::string sourceName, const int minb, const int maxb)
  {
    boost::shared_ptr<edm::PileUp> pileup; // value to be returned
    // Make sure we have a parameter named 'sourceName'
    vector<string> names = ps.getParameterNames();
    if (find(names.begin(), names.end(), sourceName)
	!= names.end())
      {
	// We have the parameter
	// and if we have either averageNumber or cfg by luminosity... make the PileUp
        double averageNumber;
        edm::ParameterSet psin=ps.getParameter<edm::ParameterSet>(sourceName);
        vector<string> namesIn = psin.getParameterNames();
        if (find(namesIn.begin(), namesIn.end(), std::string("nbPileupEvents"))
	    != namesIn.end()) {
	  edm::ParameterSet psin_average=psin.getParameter<edm::ParameterSet>("nbPileupEvents");
	  vector<string> namesAverage = psin_average.getParameterNames();
	  if (find(namesAverage.begin(), namesAverage.end(), std::string("averageNumber"))
	      != namesAverage.end()) 
	    {
              averageNumber=psin_average.getParameter<double>("averageNumber");
	      pileup.reset(new edm::PileUp(ps.getParameter<edm::ParameterSet>(sourceName),minb,maxb,averageNumber));
	      edm::LogInfo("MixingModule") <<" Created source "<<sourceName<<" with minBunch,maxBunch "<<minb<<" "<<maxb<<" and averageNumber "<<averageNumber;
	    }
	
	  //special for pileup input
	  else if (sourceName=="input" && find(namesAverage.begin(), namesAverage.end(), std::string("Lumi")) 
		   != namesAverage.end() && find(namesAverage.begin(), namesAverage.end(), std::string("sigmaInel"))
		   != namesAverage.end()) {
	    averageNumber=psin_average.getParameter<double>("Lumi")*psin_average.getParameter<double>("sigmaInel")*ps.getParameter<int>("bunchspace")/1000*3564./2808.;
	    pileup.reset(new edm::PileUp(ps.getParameter<edm::ParameterSet>(sourceName),minb,maxb,averageNumber));
	    edm::LogInfo("MixingModule") <<" Created source "<<sourceName<<" with minBunch,maxBunch "<<minb<<" "<<maxb;
	    edm::LogInfo("MixingModule")<<" Luminosity configuration, average number used is "<<averageNumber;
	  }
	}
      }

    return pileup;
  }
}

namespace edm {

  // Constructor 
  BMixingModule::BMixingModule(const edm::ParameterSet& pset) :
    bunchSpace_(pset.getParameter<int>("bunchspace")),
    checktof_(pset.getUntrackedParameter<bool>("checktof",true)),
    minBunch_((pset.getParameter<int>("minBunch")*25)/pset.getParameter<int>("bunchspace")),
    maxBunch_((pset.getParameter<int>("maxBunch")*25)/pset.getParameter<int>("bunchspace")),
    input_(maybeMakePileUp(pset,"input",minBunch_,maxBunch_)),
    beamHalo_(maybeMakePileUp(pset,"beamhalo",minBunch_,maxBunch_)),
    cosmics_(maybeMakePileUp(pset,"cosmics",minBunch_,maxBunch_)),
    md_()
  {
    md_.parameterSetID_ = pset.id();
    md_.moduleName_ = pset.getParameter<std::string>("@module_type");
    md_.moduleLabel_ = pset.getParameter<std::string>("@module_label");
    //#warning process name is hard coded, for now.  Fix this.
    //#warning the parameter set ID passed should be the one for the full process.  Fix this.
    md_.processConfiguration_ = ProcessConfiguration("PILEUP", pset.id(), getReleaseVersion(), getPassID());

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
    std::vector<EventPrincipalVector> pileup[maxNbSources];
    bool doit[]={false,false,false};

    if ( input_)  {
      LogDebug("MixingModule") <<"\n\n==============================>Adding pileup to signal event "<<e.id(); 
      input_->readPileUp(pileup[0]); 
      if ( input_->doPileup()) doit[0]=true;
    }
    if (beamHalo_) {
      beamHalo_->readPileUp(pileup[1]);
      LogDebug("MixingModule") <<"\n\n==============================>Adding beam halo to signal event "<<e.id();
      if (beamHalo_->doPileup()) doit[1]=true;
    }
    if (cosmics_) {
      cosmics_->readPileUp(pileup[2]);
      LogDebug("MixingModule") <<"\n\n==============================>Adding cosmics to signal event "<<e.id(); 
      if (cosmics_->doPileup()) doit[2]=true;
    }

    // and merge it
    // we have to loop over bunchcrossings first since added objects are all stored in one vector, 
    // ordered by bunchcrossing
    for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
      setBcrOffset();
      for (unsigned int isource=0;isource<maxNbSources;++isource) {
	if (doit[isource])   {
          setSourceOffset(isource);
	  merge(bunchCrossing, (pileup[isource])[bunchCrossing-minBunch_]);
	}	
      }
    }

    // Put output into event
    put(e);
  }

 
  void BMixingModule::merge(const int bcr, const EventPrincipalVector& vec) {
    //
    // main loop: loop over events and merge 
    //
    eventId_=0;
    LogDebug("MixingModule") <<"For bunchcrossing "<<bcr<<", "<<vec.size()<< " events will be merged";
    vertexoffset=0;
    for (EventPrincipalVector::const_iterator it = vec.begin(); it != vec.end(); ++it) {
      Event e(**it, md_);
      LogDebug("MixingModule") <<" merging Event:  id " << e.id();
      addPileups(bcr, &e, ++eventId_);
    }// end main loop
  }

} //edm
