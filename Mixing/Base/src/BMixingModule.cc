// File: BMixingModule.cc
// Description:  see BMixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
//
//--------------------------------------------

#include "Mixing/Base/interface/BMixingModule.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace std;

int edm::BMixingModule::vertexoffset = 0;
const unsigned int edm::BMixingModule::maxNbSources_ =5;

namespace
{
  boost::shared_ptr<edm::PileUp>
  maybeMakePileUp(edm::ParameterSet const& ps,std::string sourceName, const int minb, const int maxb, const bool playback)
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
        if (psin.getParameter<std::string>("type")!="none") {
	  vector<string> namesIn = psin.getParameterNames();
	  if (find(namesIn.begin(), namesIn.end(), std::string("nbPileupEvents"))
	      != namesIn.end()) {
	    edm::ParameterSet psin_average=psin.getParameter<edm::ParameterSet>("nbPileupEvents");
	    vector<string> namesAverage = psin_average.getParameterNames();
	    if (find(namesAverage.begin(), namesAverage.end(), std::string("averageNumber"))
		!= namesAverage.end()) 
	      {
		averageNumber=psin_average.getParameter<double>("averageNumber");
		pileup.reset(new edm::PileUp(ps.getParameter<edm::ParameterSet>(sourceName),minb,maxb,averageNumber,playback));
		edm::LogInfo("MixingModule") <<" Created source "<<sourceName<<" with minBunch,maxBunch "<<minb<<" "<<maxb<<" and averageNumber "<<averageNumber;
	      }
	
	    //special for pileup input
	    else if (sourceName=="input" && find(namesAverage.begin(), namesAverage.end(), std::string("Lumi")) 
		     != namesAverage.end() && find(namesAverage.begin(), namesAverage.end(), std::string("sigmaInel"))
		     != namesAverage.end()) {
	      averageNumber=psin_average.getParameter<double>("Lumi")*psin_average.getParameter<double>("sigmaInel")*ps.getParameter<int>("bunchspace")/1000*3564./2808.;  //FIXME
	      pileup.reset(new edm::PileUp(ps.getParameter<edm::ParameterSet>(sourceName),minb,maxb,averageNumber,playback));
	      edm::LogInfo("MixingModule") <<" Created source "<<sourceName<<" with minBunch,maxBunch "<<minb<<" "<<maxb;
	      edm::LogInfo("MixingModule")<<" Luminosity configuration, average number used is "<<averageNumber;
	    }
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
    checktof_(pset.getParameter<bool>("checktof")),
    minBunch_((pset.getParameter<int>("minBunch")*25)/pset.getParameter<int>("bunchspace")),
    maxBunch_((pset.getParameter<int>("maxBunch")*25)/pset.getParameter<int>("bunchspace")),
    mixProdStep1_(pset.getParameter<bool>("mixProdStep1")),
    mixProdStep2_(pset.getParameter<bool>("mixProdStep2"))	
  {
    // FIXME: temporary to keep bwds compatibility for cfg files
    vector<string> names = pset.getParameterNames();
    if (find(names.begin(), names.end(),"playback")
	!= names.end()) {
      playback_=pset.getUntrackedParameter<bool>("playback");
    } else
      playback_=false;

    //We use std::cout in order to make sure the message appears in all possible configurations of the Message Logger
    if (playback_) {
      LogInfo("MixingModule") <<" ATTENTION:Mixing will be done in playback mode! \n"
                              <<" ATTENTION:Mixing Configuration must be the same as for the original mixing!";
    }

    input_=     maybeMakePileUp(pset,"input",minBunch_,maxBunch_,playback_);
    cosmics_=   maybeMakePileUp(pset,"cosmics",minBunch_,maxBunch_,playback_);
    beamHalo_p_=maybeMakePileUp(pset,"beamhalo_plus",minBunch_,maxBunch_,playback_);
    beamHalo_m_=maybeMakePileUp(pset,"beamhalo_minus",minBunch_,maxBunch_,playback_);
    fwdDet_=maybeMakePileUp(pset,"forwardDetectors",minBunch_,maxBunch_,playback_);

    //prepare playback info structures
    fileSeqNrs_.resize(maxBunch_-minBunch_+1);
    eventIDs_.resize(maxBunch_-minBunch_+1);
    nrEvents_.resize(maxBunch_-minBunch_+1);
  }

  // Virtual destructor needed.
  BMixingModule::~BMixingModule() {;}

  // Functions that get called by framework every event
  void BMixingModule::produce(edm::Event& e, const edm::EventSetup& setup) { 
    
    // Check if the signal is present in the root file 
    // for all the objects we want to mix
    //checkSignal(e);
    
    // Create EDProduct
    createnewEDProduct();

    // Add signals
    if (!mixProdStep1_){ 
      addSignals(e,setup);
    }

    // Read the PileUp 
    for (unsigned int is=0;is< maxNbSources_;++is) {
      doit_[is]=false;
      pileup_[is].clear();
    }

    if ( input_)  {  
      if (playback_) {
	getEventStartInfo(e,0);
	input_->readPileUp(pileup_[0],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	input_->readPileUp(pileup_[0],eventIDs_, fileSeqNrs_, nrEvents_); 
        setEventStartInfo(0);
      }
      if (input_->doPileup()) {  
	LogDebug("MixingModule") <<"\n\n==============================>Adding pileup to signal event "<<e.id(); 
	doit_[0]=true;
      } 
    }
    if (cosmics_) {
      if (playback_) {
	getEventStartInfo(e,1);
	cosmics_->readPileUp(pileup_[1],eventIDs_, fileSeqNrs_, nrEvents_); 
      } else {
	cosmics_->readPileUp(pileup_[1],eventIDs_, fileSeqNrs_, nrEvents_); 
	setEventStartInfo(1);
      }
      if (cosmics_->doPileup()) {  
	LogDebug("MixingModule") <<"\n\n==============================>Adding cosmics to signal event "<<e.id(); 
	doit_[1]=true;
      } 
    }

    if (beamHalo_p_) {
      if (playback_) {
	getEventStartInfo(e,2);
	beamHalo_p_->readPileUp(pileup_[2],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	beamHalo_p_->readPileUp(pileup_[2],eventIDs_, fileSeqNrs_, nrEvents_);
	setEventStartInfo(2);
      }
      if (beamHalo_p_->doPileup()) {  
	LogDebug("MixingModule") <<"\n\n==============================>Adding beam halo+ to signal event "<<e.id();
	doit_[2]=true;
      } 
    }

    if (beamHalo_m_) {
      if (playback_) {
	getEventStartInfo(e,3);
	beamHalo_m_->readPileUp(pileup_[3],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	beamHalo_m_->readPileUp(pileup_[3],eventIDs_, fileSeqNrs_, nrEvents_);
	setEventStartInfo(3);
      }
      if (beamHalo_m_->doPileup()) {  
	LogDebug("MixingModule") <<"\n\n==============================>Adding beam halo- to signal event "<<e.id();
	doit_[3]=true;
      }
    }

    if (fwdDet_) {
      if (playback_) {
	getEventStartInfo(e,4);
	fwdDet_->readPileUp(pileup_[4],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	fwdDet_->readPileUp(pileup_[4],eventIDs_, fileSeqNrs_, nrEvents_);
	setEventStartInfo(4);
      }

      if (fwdDet_->doPileup()) {  
	LogDebug("MixingModule") <<"\n\n==============================>Adding fwd detector source  to signal event "<<e.id();
	doit_[4]=true;
      }  
    }

    doPileUp(e,setup);

    // Put output into event (here only playback info)
    put(e,setup);
  }

  void BMixingModule::merge(const int bcr, const EventPrincipalVector& vec, unsigned int worker, const edm::EventSetup& setup) {
    //
    // main loop: loop over events and merge 
    //    
    eventId_=0;
    LogDebug("MixingModule") <<"For bunchcrossing "<<bcr<<", "<<vec.size()<<" events will be merged";
    vertexoffset=0;
    for (EventPrincipalVector::const_iterator it = vec.begin(); it != vec.end(); ++it) {
      LogDebug("MixingModule") <<" merging Event:  id " << (*it)->id();
      
      addPileups(bcr, &(**it), ++eventId_,worker,setup);
    }// end main loop
  }

  void BMixingModule::dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
      if (input_) input_->dropUnwantedBranches(wantedBranches);
      if (cosmics_) cosmics_->dropUnwantedBranches(wantedBranches);
      if (beamHalo_p_) beamHalo_p_->dropUnwantedBranches(wantedBranches);
      if (beamHalo_m_) beamHalo_m_->dropUnwantedBranches(wantedBranches);
      if (fwdDet_) fwdDet_->dropUnwantedBranches(wantedBranches);
  }

  void BMixingModule::endJob() {
      if (input_) input_->endJob();
      if (cosmics_) cosmics_->endJob();
      if (beamHalo_p_) beamHalo_p_->endJob();
      if (beamHalo_m_) beamHalo_m_->endJob();
      if (fwdDet_) fwdDet_->endJob();
  }

} //edm
