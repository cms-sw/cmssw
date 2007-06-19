#ifndef AnalyzeTriggerResults_h_
#define AnalyzeTriggerResults_h_

#include <map>
#include <string>
#include <iostream>

#include <TH2F.h>
#include <TAxis.h>
#include <TFile.h>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"

// class for analyzing the trigger decisions
class AnalyzeTriggerResults
{
  
 public:
  AnalyzeTriggerResults()
  {
    Ntp = 0;
    Nall_trig = Nevents = 0;
  }
  //
  ~AnalyzeTriggerResults()
  {
    
  }
  // call this method for every event 
  void analyze(const edm::TriggerResults & trigRes)
  {
    ++Nevents;

    // get the # of trigger paths for this event
    unsigned int size = trigRes.size();
    if(Ntp)
      assert(Ntp == size);
    else
      Ntp = size;

    // has any trigger fired this event?
    if(trigRes.accept())++Nall_trig;

    edm::TriggerNames triggerNames(trigRes);

    // loop over all paths, get trigger decision
    for(unsigned i = 0; i != size; ++i)
      {
	std::string name = triggerNames.triggerName(i);
	fired[name] = trigRes.accept(i);
	if(fired[name])
	  ++(Ntrig[name]);
      }
    
    // NOTE: WE SHOULD MAKE THIS A SYMMETRIC MATRIX...
    // double-loop over all paths, get trigger overlaps
     for(unsigned i = 0; i != size; ++i)
      {
	std::string name = triggerNames.triggerName(i);
	if(!fired[name])continue;

	bool correlation = false;

	for(unsigned j = 0; j != size; ++j)
	  {
	    // skip the same name; 
	    // this entry correponds to events triggered by single trigger
	    if(i == j) continue;
	    std::string name2 = triggerNames.triggerName(j);
	    if(fired[name2])
	      {
		correlation = true;
		++(Ncross[name][name2]);
	      }
	  } // loop over j-trigger paths

	if(!correlation) // events triggered by single trigger
	  ++(Ncross[name][name]);
	  
      } //  // loop over i-trigger paths

  }
  
  // call this after all events have been processed
  void getResults()
  {
    const int Nbins = Ntp;
    h2_cross = new TH2F("trigger_rates", "Trigger rates", Nbins, 0, 
			    Nbins, Nbins, 0, Nbins);
    h2_cross->SetDirectory(0);
 /*    h2_cross->GetXaxis()->SetTitle("(module)"); */
/*     h2_cross->GetYaxis()->SetTitle("(module)"); */

    std::cout << " Total trigger rate: " << Nall_trig << "/" << Nevents
	      << " events (" << 100.*Nall_trig/Nevents << "%) " 
	      << std::endl << std::endl;

    std::cout << " Individual path rates: " << std::endl;
    typedef trigPath::iterator It;
    int ix = 1;
    for(It it = Ntrig.begin(); 
	it != Ntrig.end(); ++it, ++ix)
      {
	std::cout << " Trigger path \"" << it->first << "\": " 
		  << it->second << "/"
		  << Nevents << " events (" << 100.*(it->second)/Nevents 
		  << "%) " << std::endl;
	std::cout << std::endl;

	h2_cross->GetXaxis()->SetBinLabel(ix, it->first.c_str());
	h2_cross->GetYaxis()->SetBinLabel(ix, it->first.c_str());
      }

    std::cout << " Trigger path correlations: " << std::endl;
    typedef std::map<std::string, trigPath>::iterator IIt;

    ix = 1;
    for(IIt it = Ncross.begin(); 
	it != Ncross.end(); ++it, ++ix)
      { // loop over first trigger of pair

	trigPath & cross = it->second;
	int iy = 1;
	for(It it2 = cross.begin(); 
	    it2 != cross.end(); ++it2, ++iy)
	  { // loop over second trigger of pair

	    h2_cross->SetBinContent(ix, iy, it2->second);

	    // skip symmetric pairs: 1st pass does "path1", "path2";
	    // 2nd pass should skip "path2", "path1"
	    if(it->first > it2->first)continue;

	    // print out first trigger
	    std::cout << " \"" << it->first << "\"";

	    // if second trigger = first trigger, 
	    // this corresponds to unique rate (ie. no correlation)
	    if(it->first == it2->first)
	      std::cout << " (unique rate): ";
	    else
	      std::cout << " x \"" << it2->first << "\": ";

	    std::cout << it2->second << "/"
		 << Nevents << " events (" << 100.*(it2->second)/Nevents 
		 << "%) " << std::endl;
	  }
      }

    TFile * out_file = new TFile("HLT_triggers.root", "RECREATE");
    
    h2_cross->Write();
    //    out_file->Write();
    out_file->Close();
    delete out_file;
    delete h2_cross;
    
  }

 private:
  // event counters
  unsigned int Ntp; // # of trigger paths (should be the same for all events!)
  unsigned int Nall_trig; // # of all triggered events
  unsigned int Nevents; // # of analyzed events
  
  typedef std::map<std::string, unsigned int> trigPath;

  trigPath Ntrig; // # of triggered events per path

  // # of cross-triggered events per path
  // (pairs with same name correspond to unique trigger rates for that path)
  std::map<std::string, trigPath> Ncross;

  // whether a trigger path has fired for given event
  // (variable with event-scope)
  std::map<std::string, bool> fired; 

  TH2F * h2_cross;
};


#endif // #define AnalyzeTriggerResults_h_
