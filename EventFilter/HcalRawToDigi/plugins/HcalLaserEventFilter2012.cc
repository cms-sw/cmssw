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
#include "zlib.h"

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

  void readEventListFile(const std::string & eventFileName);
  void addEventString(const std::string & eventString);

 // ----------member data ---------------------------
  typedef std::vector< std::string > strVec;
  typedef std::vector< std::string >::iterator strVecI;
  
  std::vector< std::string > EventList_;  // vector of strings representing bad events, with each string in "run:LS:event" format
  bool verbose_;  // if set to true, then the run:LS:event for any event failing the cut will be printed out
  std::string prefix_;  // prefix will be printed before any event if verbose mode is true, in order to make searching for events easier
  
  // Set run range of events in the BAD LASER LIST.  
  // The purpose of these values is to shorten the length of the EventList_ vector when running on only a subset of data
  int minrun_;
  int maxrun_;  // if specified (i.e., values > -1), then only events in the given range will be filtered
  int minRunInFile, maxRunInFile;

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
using namespace std;
#define CHUNK 16384

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

  minRunInFile=999999; maxRunInFile=1;
  string eventFileName=ps.getParameter<string>("eventFileName");
  if (verbose_) edm::LogInfo("HcalLaserHFFilter2012") << "HCAL laser event list from file "<<eventFileName;
  readEventListFile(eventFileName);
  std::sort(EventList_.begin(), EventList_.end());
  if (verbose_) edm::LogInfo("HcalLaserHFFilter2012")<<" A total of "<<EventList_.size()<<" listed HCAL laser events found in given run range";
  if (minrun_==-1 || minrun_<minRunInFile) minrun_=minRunInFile;
  if (maxrun_==-1 || maxrun_>maxRunInFile) maxrun_=maxRunInFile;
}

void HcalLaserEventFilter2012::addEventString(const string & eventString)
{
  // Loop through list of bad events, and if run is in allowed range, add bad event to EventList
  int run=0;
  unsigned int ls=0;
  unsigned int event=0;
  // Check that event list object is in correct form
  size_t found = eventString.find(":");  // find first colon
  if (found!=std::string::npos)
    run=atoi((eventString.substr(0,found)).c_str());  // convert to run
  else
    {
      edm::LogError("HcalLaserHFFilter2012")<<"  Unable to parse Event list input '"<<eventString<<"' for run number!";
      return;
    }
  size_t found2 = eventString.find(":",found+1);  // find second colon
  if (found2!=std::string::npos)
    {
      /// Some event numbers are less than 0?  \JetHT\Run2012C-v1\RAW:201278:2145:-2130281065  -- due to events being dumped out as ints, not uints!
      ls=atoi((eventString.substr(found+1,(found2-found-1))).c_str());  // convert to ls
      event=atoi((eventString.substr(found2+1)).c_str()); // convert to event
      /// Some event numbers are less than 0?  \JetHT\Run2012C-v1\RAW:201278:2145:-2130281065
      if (ls==0 || event==0) edm::LogWarning("HcalLaserHFFilter2012")<<"  Strange lumi, event numbers for input '"<<eventString<<"'";
    }
  else
    {
      edm::LogError("HcalLaserHFFilter2012")<<"Unable to parse Event list input '"<<eventString<<"' for run number!";
      return;
    }
  // If necessary, check that run is within allowed range
  if (minrun_>-1 && run<minrun_)
    {
      if (verbose_)  edm::LogInfo("HcalLaserHFFilter2012") <<"Skipping Event list input '"<<eventString<<"' because it is less than minimum run # "<<minrun_;
      return;
    }
  if (maxrun_>-1 && run>maxrun_)
    {
      if (verbose_) edm::LogInfo("HcalLaserHFFilter2012") <<"Skipping Event list input '"<<eventString<<"' because it is greater than maximum run # "<<maxrun_;
      return;
    }
  if (minRunInFile>run) minRunInFile=run;
  if (maxRunInFile<run) maxRunInFile=run;
  // Now add event to Event List
  EventList_.push_back(eventString);
}

#define LENGTH 0x2000

void HcalLaserEventFilter2012::readEventListFile(const string & eventFileName)
{
  gzFile  file = gzopen (eventFileName.c_str(), "r");
  if (! file) {
    edm::LogError("HcalLaserHFFilter2012")<<"  Unable to open event list file "<<eventFileName;
    return;
  }
  string b2;
  int err;                    
  int bytes_read;
  char buffer[LENGTH];
  unsigned int i;
  char * pch;

  while (1) {
    bytes_read = gzread (file, buffer, LENGTH - 1);
    buffer[bytes_read] = '\0';
    i=0;
    pch = strtok (buffer,"\n");
    if (buffer[0] == '\n' ) {
      addEventString(b2);
      ++i;
    } else b2+=pch;
  
    while (pch != NULL)
    {
      i+=strlen(pch)+1;
      if (i>b2.size()) b2= pch;
      if (i==(LENGTH-1)) {
	 if ((buffer[LENGTH-2] == '\n' )|| (buffer[LENGTH-2] == '\0' )){
          addEventString(b2);
          b2="";
	}
      } else if (i<LENGTH) {
	addEventString(b2);
      } 
      pch = strtok (NULL, "\n");
    }
    if (bytes_read < LENGTH - 1) {
      if (gzeof (file)) break;
        else {
          const char * error_string;
          error_string = gzerror (file, & err);
          if (err) {
	    edm::LogError("HcalLaserHFFilter2012")<<"Error while reading gzipped file:  "<<error_string;
            return;
          }
        }
    }
  }
  gzclose (file);
  return;
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
  strVecI it = std::lower_bound(EventList_.begin(), EventList_.end(), thisevent.str());
  if (it == EventList_.end() || thisevent.str() < *it) return true;
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
