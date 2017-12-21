#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TSystem.h>

#include <iostream>
#include <string>
#include <assert.h>
#include <stdlib.h>

// needed for timing studies
#include "DataFormats/HLTReco/interface/ModuleTiming.h"
// needed for event-id info
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
//
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

#include "HLTrigger/Timer/test/AnalyzeTiming.h"

using std::cout; using std::endl; using std::string;

#define PRINT_EVTRUN_NO 0

// Christos Leonidopoulos, July 2006


/* usage: analyzeTiming <ROOT file> <process name> <N_bins> <Max_time>
   where, 
   <ROOT file>: file with HLT information to be analyzed (default: hlt.root)
   <process name>: name for cmsRun job in config file (default: PRODHLT)
   <N_bins>: # of bins for timing histograms (default: 100)
   <Max_time>: upper edge for timing histograms in ms (default: 1000)
*/
int main(int argc, char ** argv)
{
  // load libraries
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();

  // default arguments
  string filename = "hlt.root";
  string process_name = "PRODHLT";
  unsigned int N_bins = 100;
  float Max_time = 1000; // in ms
  // get parameters from command line
  if(argc >= 2) filename = argv[1];
  if(argc >= 3) process_name = argv[2];
  if(argc >= 4) N_bins = atoi(argv[3]);
  if(argc >= 5) Max_time = 1.*atoi(argv[4]);

  // open file
  TFile file(filename.c_str());
  if(file.IsZombie()) 
    {
      cout << " *** Error opening file " << filename << endl;
      exit(-1);
    }
  TTree * events = dynamic_cast<TTree *>(file.Get("Events") );
  assert(events);

  TBranch * TBevtTime = 0;
  TBranch * TBevtAux = events->GetBranch("EventAuxiliary");
  assert(TBevtAux);
  //  std::cout << " TBevtAux = " << int(TBevtAux) << std::endl;

  // structure holding the timing info
  edm::EventTime evtTime;

#if PRINT_EVTRUN_NO
  // structure holding event information
  edm::EventAuxiliary * evtAux = new edm::EventAuxiliary;

  TBevtAux->SetAddress((void *)&evtAux);
#endif

  AnalyzeTiming * tt = 0;

  char tmp_name[1024];
  snprintf(tmp_name, 1024, "edmEventTime_myTimer__%s.obj", process_name.c_str());
  TBevtTime = events->GetBranch(tmp_name);
  assert(TBevtTime);
  TBevtTime->SetAddress((void*)&evtTime);
  tt = new AnalyzeTiming(N_bins, 0, Max_time);

  int n_evts = events->GetEntries();
  
  for(int i = 0; i != n_evts; ++i)
    {

#if PRINT_EVTRUN_NO
      TBevtAux->GetEntry(i);
      cout << " Run # = " << evtAux->id().run() 
	   << " event # = " << evtAux->id().event() 
	   << " entry # = " << i << "\n";
#endif

      TBevtTime->GetEntry(i);
      tt->analyze(evtTime);

    } // loop over all events


  // get results, do cleanup

  tt->getResults();
  delete tt;

#if PRINT_EVTRUN_NO
  delete evtAux;
#endif
 
  return 0;
}
