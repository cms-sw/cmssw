#ifndef PhysicsTools_Utilities_EventFilterFromList
#define PhysicsTools_Utilities_EventFilterFromList

// -*- C++ -*-
//
// Package:    PhysicsTools
// Class:      EventFilterFromListStandAlone
//
// $Id: EventFilterFromListStandAlone.h,v 1.3 2013/01/16 16:42:48 vadler Exp $
//
/**
  \class    EventFilterFromListStandAlone EventFilterFromListStandAlone.h "physicsTools/utilities/interface/EventFilterFromListStandAlone.h"
  \brief    Stand-alone class to flag events, based on an event list in a gzipped tex file

   EventFilterFromListStandAlone provides a boolean flag, which marks events as "bad", if they appear in a given gzipped text file.
   The boolean returns
   - 'false', if the event is in the list ("bad event") and
   - 'true' otherwise ("good event").

   The class is designed as stand-alone utility, so that it can be used outside CMSSW, too.
   It is instatiated as follows:

     #include "[PATH/]EventFilterFromListStandAlone.h"
     [...]
     EventFilterFromListStandAlone myFilter("[GZIPPED_TEXT_FILE]");

   !!! --->>> There might be OFFICIAL releases of such EVENT LISTS, provided by the PdmV group <<<--- !!!
   An important example is the list of HCAL laser polluted events in

   EventFilter/HcalRawToDigi/data/HCALLaser2012AllDatasets.txt.gz

   The path to the gzipped input file needs to be the real path. CMSSW-like "FileInPath" is not supported.

   The boolean is then determined with

     bool myFlag = myFilter.filter(run_number, lumi_sec_number, event_number);

   where the parameters are all of type 'int'.

   Compilation:
   ============

   EventFilterFromListStandAlone uses 'zlib', which requires varying compilation settings, depending on the environment:

   LXPLUS, no CMSSW:
   -----------------
   - Files needed in current directory:
     EventFilterFromListStandAlone.h # a copy of this file
     test.cc                         # the actual code, using this class
     events.txt.gz                   # gzipped input file
   - In test.cc, you have
       #include "EventFilterFromListStandAlone.h"
       [...]
       EventFilterFromListStandAlone myFilter("./events.txt.gz");
       [...]
       bool myFlag = myFilter.filter(run_number, lumi_sec_number, event_number)
   - To compile:
       source /afs/cern.ch/sw/lcg/contrib/gcc/4.3/x86_64-slc5/setup.[c]sh
       source /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.00/x86_64-slc5-gcc43-opt/root/bin/thisroot.[c]sh
       g++ -I$ROOTSYS/include -L$ROOTSYS/lib -lCore test.cc
     which results in the default executable './a.out'.

   LXPLUS, CMSSW environment
   -------------------------
   - Files needed in current directory:
     test.cc                         # the actual code, using this class
     events.txt.gz                   # gzipped input file
                                     # e.g. by
                                     # cp $CMSSW_RELEASE_BASE/src/EventFilter/HcalRawToDigi/data/HCALLaser2012AllDatasets.txt.gz ./events.txt.gz
   - In test.cc, you have
       #include "PhysicsTools/Utilities/interface/EventFilterFromListStandAlone.h"
       [...]
       EventFilterFromListStandAlone myFilter("./events.txt.gz");
       [...]
       bool myFlag = myFilter.filter(run_number, lumi_sec_number, event_number)
   - To compile:
       g++ -I$CMS_PATH/$SCRAM_ARCH/external/zlib/include -L$CMS_PATH/$SCRAM_ARCH/external/zlib/lib -lz test.cc
     which results in the default executable './a.out'.

   LXPLUS, CMSSW environment, compilation with SCRAM
   -------------------------------------------------
   - Files needed in code directory (e.g. $CMSSW_BASE/src/[SUBSYSTEM]/[PACKAGE]/bin/):
     test.cc                         # the actual code, using this class
     BuildFile.xml                   #
   - Files needed in current directory:
     events.txt.gz                   # gzipped input file
                                     # e.g. by
                                     # cp $CMSSW_RELEASE_BASE/src/EventFilter/HcalRawToDigi/data/HCALLaser2012AllDatasets.txt.gz ./events.txt.gz
   - In test.cc, you have
       #include "PhysicsTools/Utilities/interface/EventFilterFromListStandAlone.h"
       [...]
       EventFilterFromListStandAlone myFilter("./events.txt.gz");
       [...]
       bool myFlag = myFilter.filter(run_number, lumi_sec_number, event_number)
   - In BuildFile.xml, you have:
       <use name="zlib"/>
       <use name="PhysicsTools/Utilities"/>
       [...]
       <environment>
        [...]
        <bin file="test.cc"></bin>
        [...]
       </environment>
   - To compile:
       scram b # evtl. followed by 'rehash' (for csh) to make the executable available
     which results in the executable '$CMSSW_BASE/bin/$SCRAM_ARCH/test'.

  \author   Thomas Speer
  \version  $Id: EventFilterFromListStandAlone.h,v 1.3 2013/01/16 16:42:48 vadler Exp $
*/

#include <vector>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include "zlib.h"
#include <stdio.h>

class EventFilterFromListStandAlone  {
public:
  /**
   * Constructor
   * eventFileName: The gzipped file with the list of events
   */

  EventFilterFromListStandAlone(const std::string & eventFileName);

  ~EventFilterFromListStandAlone() {}

  /**
   * The filter, returns true for good events, bad for events which are
   * to be filtered out, i.e. events that are in the list
   */
  bool filter(int run, int lumiSection, int event);

private:

  void readEventListFile(const std::string & eventFileName);
  void addEventString(const std::string & eventString);

  typedef std::vector< std::string > strVec;
  typedef std::vector< std::string >::iterator strVecI;

  std::vector< std::string > EventList_;  // vector of strings representing bad events, with each string in "run:LS:event" format
  bool verbose_;  // if set to true, then the run:LS:event for any event failing the cut will be printed out

  // Set run range of events in the BAD event LIST.
  // The purpose of these values is to shorten the length of the EventList_ vector when running on only a subset of data
  int minrun_;
  int maxrun_;  // if specified (i.e., values > -1), then only events in the given range will be filtered
  int minRunInFile, maxRunInFile;

};

using namespace std;

EventFilterFromListStandAlone::EventFilterFromListStandAlone(const std::string & eventFileName)
{
  verbose_=false;
  minRunInFile=999999; maxRunInFile=1;
  minrun_ = 0;  maxrun_ = 999999;
  if (verbose_) cout << "Event list from file "<<eventFileName<<endl;
  readEventListFile(eventFileName);
  std::sort(EventList_.begin(), EventList_.end());
  if (verbose_) cout<<" A total of "<<EventList_.size()<<" listed events found in given run range"<<endl;
  minrun_=minRunInFile;
  maxrun_=maxRunInFile;
}

void EventFilterFromListStandAlone::addEventString(const string & eventString)
{
  int run=0;
  unsigned int ls=0;
  unsigned int event=0;
  // Check that event list object is in correct form
  size_t found = eventString.find(":");  // find first colon
  if (found!=std::string::npos)
    run=atoi((eventString.substr(0,found)).c_str());  // convert to run
  else
    {
      cout<<"  Unable to parse Event list input '"<<eventString<<"' for run number!\n";
      return;
    }
  size_t found2 = eventString.find(":",found+1);  // find second colon
  if (found2!=std::string::npos)
    {
      /// Some event numbers are less than 0?  \JetHT\Run2012C-v1\RAW:201278:2145:-2130281065  -- due to events being dumped out as ints, not uints!
      ls=atoi((eventString.substr(found+1,(found2-found-1))).c_str());  // convert to ls
      event=atoi((eventString.substr(found2+1)).c_str()); // convert to event
      /// Some event numbers are less than 0?  \JetHT\Run2012C-v1\RAW:201278:2145:-2130281065
      if (ls==0 || event==0) cout<<"  Strange lumi, event numbers for input '"<<eventString<<"'";
    }
  else
    {
      cout<<"Unable to parse Event list input '"<<eventString<<"' for run number!\n";
      return;
    }
  // If necessary, check that run is within allowed range
  if (minrun_>-1 && run<minrun_)
    {
      if (verbose_)  cout <<"Skipping Event list input '"<<eventString<<"' because it is less than minimum run # "<<minrun_<<endl;
      return;
    }
  if (maxrun_>-1 && run>maxrun_)
    {
      if (verbose_) cout <<"Skipping Event list input '"<<eventString<<"' because it is greater than maximum run # "<<maxrun_<<endl;
      return;
    }
  if (minRunInFile>run) minRunInFile=run;
  if (maxRunInFile<run) maxRunInFile=run;
  // Now add event to Event List
  EventList_.push_back(eventString);
}

#define LENGTH 0x2000

void EventFilterFromListStandAlone::readEventListFile(const string & eventFileName)
{
  gzFile  file = gzopen (eventFileName.c_str(), "r");
  if (! file) {
    cout<<"  Unable to open event list file "<<eventFileName<<endl;
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
	    cout<<"Error while reading gzipped file:  "<<error_string<<endl;
            return;
          }
        }
    }
  }
  gzclose (file);
  return;
}

bool
EventFilterFromListStandAlone::filter(int run, int lumiSection, int event)
{
  // if run is outside filter range, then always return true
  if (minrun_>-1 && run<minrun_) return true;
  if (maxrun_>-1 && run>maxrun_) return true;

  // Okay, now create a string object for this run:ls:event
  std::stringstream thisevent;
  thisevent<<run<<":"<<lumiSection<<":"<<event;

  // Event not found in bad list; it is a good event
  strVecI it = std::lower_bound(EventList_.begin(), EventList_.end(), thisevent.str());
  if (it == EventList_.end() || thisevent.str() < *it) return true;
  // Otherwise, this is a bad event
  // if verbose, dump out event info
  // Dump out via cout, or via LogInfo?  For now, use cout
  if (verbose_) std::cout <<"EventFilterFromListStandAlone removed "<<thisevent.str()<<std::endl;

  return false;
}

#endif
