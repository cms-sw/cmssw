#ifndef AnalyzeTiming_h_
#define AnalyzeTiming_h_

#include <TH1F.h>
#include <TFile.h>

#include "DataFormats/HLTReco/interface/ModuleTiming.h"

#include <map>
#include <string>
#include <iostream>

// class for analyzing the timing of the trigger modules
class AnalyzeTiming
{
 private:
  // times are given in secs; convert to msecs here
  static const double secs2msecs;

 public:
  // create object with # of bins and range for histogram (in msecs)
  AnalyzeTiming(unsigned NBINS, double Xlow, double Xmax)
    : tot_time(0)
  {
    Nbins = NBINS; xlow = Xlow; xmax = Xmax;
    moduleTimes.clear();
    tot_time = createHistogram("full_event");    
  }
  //
  ~AnalyzeTiming()
  {
    if(tot_time)delete tot_time;

    for(cIt it = moduleTimes.begin(); it != moduleTimes.end(); ++it)
      delete it->second;
    moduleTimes.clear();
  }

  // call this method for every event 
  void analyze( const edm::EventTime & evtTime)
  {
    unsigned size = evtTime.size();
    for(unsigned int i = 0; i != size; ++i)
      { // loop over all modules of event

	std::string module_name = evtTime.name(i);
	TH1F * tmp = 0;
	cIt it = moduleTimes.find(module_name);
	if(it != moduleTimes.end())
	  tmp = it->second;
	else
	  {
	    // first time module is encountered; create new histogram
	    tmp = createHistogram(module_name);

	    // add entry to map

	    /* this for some reason creates duplicate entries in moduleTimes... */
	    //	    moduleTimes[module_name] = tmp;

	    moduleTimes[module_name.c_str()] = tmp;

	  }

	tmp->Fill(evtTime.time(i)*secs2msecs);

      } // loop over all modules of event

    tot_time->Fill(evtTime.tot_time()*secs2msecs);
  }

  // call this after all events have been processed
  void getResults()
  {
    TFile * out_file = new TFile("HLT_timing.root", "RECREATE");
    for(cIt it = moduleTimes.begin(); it != moduleTimes.end(); ++it)
      (it->second)->Write();

    tot_time->Write();
    //    out_file->Write();
    out_file->Close();
    delete out_file;
  }


 private:
  TH1F * createHistogram(std::string name)
  {
    std::cout << " Creating histogram for newly added module \"" << name 
	      << "\"" << std::endl;
    char title[1024];
    snprintf(title, 1024, "Processing times for %s", name.c_str());
    TH1F * newHistogram = new TH1F(name.c_str(), title, Nbins, xlow, xmax);
    newHistogram->GetXaxis()->SetTitle("t (ms)");
    // will do our own memory management
    newHistogram->SetDirectory(0);
    return newHistogram;
  }

  // time distributions
  // (a) per module
  std::map<std::string, TH1F *> moduleTimes;
  typedef std::map<std::string, TH1F *>::const_iterator cIt;
  // (b) per event
  TH1F * tot_time;

  // histogram parameters (common for all histograms)
  unsigned int Nbins; // # of bins
  double xlow, xmax; // histogram range

};

const double AnalyzeTiming::secs2msecs = 1000;

#endif // #define AnalyzeTiming_h_
