// -*- C++ -*-
//
// Package:    HcalLaserEventFilter2012
// Class:      HcalLaserEventFilter2012
// 
/**\class HcalLaserEventFilter2012 HcalLaserEventFilter2012.cc UserCode/HcalLaserEventFilter2012/src/HcalLaserEventFilter2012.cc

 Description: [Remove known HCAL laser events in 2012 data]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jeff Temple, University of Maryland (jtemple@fnal.gov)
//         Created:  Fri Oct 19 13:15:44 EDT 2012
//
//


// system include files
#include <memory>
#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunID.h"


//
// class declaration
//

class HcalLaserEventFilter2012 : public edm::EDFilter {
public:
  explicit HcalLaserEventFilter2012(const edm::ParameterSet&);
  ~HcalLaserEventFilter2012();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  virtual bool beginRun(edm::Run&, edm::EventSetup const&);
  virtual bool endRun(edm::Run&, edm::EventSetup const&);
  virtual bool beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  virtual bool endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

 // ----------member data ---------------------------
  std::vector< std::string > EventList_;  // vector of strings representing bad events, with each string in "run:LS:event" format
  bool verbose_;  // if set to true, then the run:LS:event for any event failing the cut will be printed out
  std::string prefix_;  // prefix will be printed before any event if verbose mode is true, in order to make searching for events easier
  
  // Set run range of events in the BAD LASER LIST.  
  // The purpose of these values is to shorten the length of the EventList_ vector when running on only a subset of data
  int minrun_;
  int maxrun_;  // if specified (i.e., values > -1), then only events in the given range will be filtered

  bool WriteBadToFile_;
  bool forceFilterTrue_;
  std::ofstream outfile_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HcalLaserEventFilter2012::HcalLaserEventFilter2012(const edm::ParameterSet& ps)
{
  verbose_ = ps.getUntrackedParameter<bool>("verbose",false);
  prefix_  = ps.getUntrackedParameter<std::string>("prefix","");
  minrun_  = ps.getUntrackedParameter<int>("minrun",-1);
  maxrun_  = ps.getUntrackedParameter<int>("maxrun",-1);
  WriteBadToFile_=ps.getUntrackedParameter<bool>("WriteBadToFile",false);
  if (WriteBadToFile_)
    outfile_.open("badHcalLaserList_eventfilter.txt");
  forceFilterTrue_=ps.getUntrackedParameter<bool>("forceFilterTrue",false);

  std::vector<std::string> dummylist;
  dummylist.clear();
  std::vector<std::string> templist;
  templist=ps.getUntrackedParameter<std::vector<std::string> >("EventList",dummylist); // get list of events
  
  // Loop through list of bad events, and if run is in allowed range, add bad event to EventList
  for (unsigned int i=0;i<templist.size();++i)
    {
      int run=0;
      unsigned int ls=0;
      unsigned int event=0;
      // Check that event list object is in correct form
      size_t found = templist[i].find(":");  // find first colon
      if (found!=std::string::npos)
	run=atoi((templist[i].substr(0,found)).c_str());  // convert to run
      else
	{
	  edm::LogError("HcalLaserHFFilter2012")<<"  Unable to parse Event list input '"<<templist[i]<<"' for run number!";
	  continue;
	}
      size_t found2 = templist[i].find(":",found+1);  // find second colon
      if (found2!=std::string::npos)
	{
	  /// Some event numbers are less than 0?  \JetHT\Run2012C-v1\RAW:201278:2145:-2130281065  -- due to events being dumped out as ints, not uints!
	  ls=atoi((templist[i].substr(found+1,(found2-found-1))).c_str());  // convert to ls
	  event=atoi((templist[i].substr(found2+1)).c_str()); // convert to event
	  /// Some event numbers are less than 0?  \JetHT\Run2012C-v1\RAW:201278:2145:-2130281065
	  if (ls==0 || event==0) edm::LogWarning("HcalLaserHFFilter2012")<<"  Strange lumi, event numbers for input '"<<templist[i]<<"'";
	}
      else
	{
	  edm::LogError("HcalLaserHFFilter2012")<<"Unable to parse Event list input '"<<templist[i]<<"' for run number!";
	  continue;
	}
      // If necessary, check that run is within allowed range
      if (minrun_>-1 && run<minrun_)
	{
	  if (verbose_)  edm::LogInfo("HcalLaserHFFilter2012") <<"Skipping Event list input '"<<templist[i]<<"' because it is less than minimum run # "<<minrun_;
	  continue;
	}
      if (maxrun_>-1 && run>maxrun_)
	{
	  if (verbose_) edm::LogInfo("HcalLaserHFFilter2012") <<"Skipping Event list input '"<<templist[i]<<"' because it is greater than maximum run # "<<maxrun_;
	  continue;
	}
      // Now add event to Event List
      EventList_.push_back(templist[i]);
    }
  if (verbose_) edm::LogInfo("HcalLaserHFFilter2012")<<" A total of "<<EventList_.size()<<" listed HCAL laser events found in given run range";

}


HcalLaserEventFilter2012::~HcalLaserEventFilter2012()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HcalLaserEventFilter2012::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  int run = iEvent.id().run();
  // if run is outside filter range, then always return true
  if (minrun_>-1 && run<minrun_) return true;
  if (maxrun_>-1 && run>maxrun_) return true;

  // Okay, now create a string object for this run:ls:event
  std::stringstream thisevent;
  thisevent<<run<<":"<<iEvent.luminosityBlock()<<":"<<iEvent.id().event();

  // Event not found in bad list; it is a good event
  if (std::find(EventList_.begin(), EventList_.end(), thisevent.str())==EventList_.end())
    return true;
  // Otherwise, this is a bad event
  // if verbose, dump out event info
  // Dump out via cout, or via LogInfo?  For now, use cout
  if (verbose_) std::cout <<prefix_<<thisevent.str()<<std::endl;

  // To use if we decide on LogInfo:
  // if (verbose_) edm::LogInfo(prefix_)<<thisevent.str();
  
  // Write bad event to file 
  if (WriteBadToFile_)
    outfile_<<iEvent.id().run()<<":"<<iEvent.luminosityBlock()<<":"<<iEvent.id().event()<<std::endl;
  if (forceFilterTrue_) return true; // if special input boolean set, always return true, regardless of filter decision
  return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalLaserEventFilter2012::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalLaserEventFilter2012::endJob() {
 if (WriteBadToFile_) outfile_.close();
}

// ------------ method called when starting to processes a run  ------------
bool 
HcalLaserEventFilter2012::beginRun(edm::Run&, edm::EventSetup const&)
{ 
  return true;
}

// ------------ method called when ending the processing of a run  ------------
bool 
HcalLaserEventFilter2012::endRun(edm::Run&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when starting to processes a luminosity block  ------------
bool 
HcalLaserEventFilter2012::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when ending the processing of a luminosity block  ------------
bool 
HcalLaserEventFilter2012::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HcalLaserEventFilter2012::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalLaserEventFilter2012);
