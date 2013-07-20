// -*- C++ -*-
//
// Package:    PickEvents
// Class:      PickEvents
// 
/**\class PickEvents PickEvents.cc DPGAnalysis/PickEvents/src/PickEvents.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Henry Schmitt
//         Created:  Mon Sep 15 19:36:37 CEST 2008
// $Id: PickEvents.cc,v 1.5 2013/02/27 20:17:14 wmtan Exp $
//         Modified: 27/03/2009 Luca Malgeri
//                   reading external file, defining selection syntax
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <limits>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"


//
// class declaration
//

class PickEvents : public edm::EDFilter {
   public:
      explicit PickEvents(const edm::ParameterSet&);
      ~PickEvents();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

  std::string listrunevents_; 
  std::string listruneventsinpath_; 
  
  std::vector<bool> whattodo;
  std::vector<edm::RunNumber_t> startrun;
  std::vector<edm::RunNumber_t> endrun;
  std::vector<edm::EventNumber_t> startevent;
  std::vector<edm::EventNumber_t> endevent;
  
  int nEventsAnalyzed;
  int nEventsSelected;
      
};



PickEvents::PickEvents(const edm::ParameterSet& iConfig)
{

  listruneventsinpath_=iConfig.getUntrackedParameter<std::string> ("RunEventList","");
  edm::FileInPath listruneventstmp("DPGAnalysis/Skims/data/"+listruneventsinpath_);

  listrunevents_=listruneventstmp.fullPath();

  std::cout <<"File with run/event list:"<< listrunevents_<<std::endl;

}


PickEvents::~PickEvents()
{
}

bool
PickEvents::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   RunNumber_t kRun   = iEvent.id().run();
   EventNumber_t kEvent = iEvent.id().event();

   bool selectThisEvent = false;

   for (unsigned int cond=0; cond<whattodo.size();cond++)
     {
       //       std::string what;
       if ( kRun>=startrun[cond] && 
	    kRun<=endrun[cond]   &&
	    kEvent>=startevent[cond] && 
	    kEvent<=endevent[cond] ) 
	 { // it's in the range, use
	   selectThisEvent=whattodo[cond];
	 }
     }
	       
   nEventsAnalyzed++;
   if (selectThisEvent) nEventsSelected++;
   //   if (selectThisEvent) std::cout << "Event selected: " << kRun << " " << kEvent << std::endl;

   return selectThisEvent;
}

void 
PickEvents::beginJob()
{
  using namespace std;

  std::string line;
  std::string buf;

  std::stringstream ss;
  std::vector<std::string> tokens;

  nEventsAnalyzed = 0;
  nEventsSelected = 0;

  // open file listevent file
  ifstream listfile;
  listfile.open(listrunevents_.c_str());
  if (listfile.is_open())
    {
      while (! listfile.eof() )
	{
	  getline (listfile,line);
	  ss.clear();
	  ss.str(line);
	  tokens.clear();
	  while (ss>>buf)
	    {
	      tokens.push_back(buf);
	      //	      std::cout << buf << std::endl;
	    }	      
	  // std::cout << tokens.size() << std::endl;
	  if (tokens.size()<3)
	    {
	      //	      std::cout << "strange selection line:" << line << std::endl;
	      //	      std::cout << "skipping it" << std::endl;
	      continue;
	    }
	  if(tokens[0]=="-" || tokens[0]=="+")
	    {
	      // it's a selection line, use it
	      if(tokens[0]=="-") whattodo.push_back(false);
	      else whattodo.push_back(true);

	      // start with run selecion
	      int loc=tokens[1].find(":",0);

	      std::string first=tokens[1].substr(0,loc);
	      startrun.push_back((edm::RunNumber_t)atoi(first.c_str()));

	      std::string last=tokens[1].substr(loc+1,tokens[1].size());
	      if (last=="infty")
		endrun.push_back(std::numeric_limits<unsigned int>::max());
	      else
		endrun.push_back((edm::RunNumber_t) atoi(last.c_str()));

	      // then event number selecion
	      loc=tokens[2].find(":",0);

	      first=tokens[2].substr(0,loc);
	      startevent.push_back((edm::EventNumber_t)atoi(first.c_str()));

	      last=tokens[2].substr(loc+1,tokens[2].size());
	      if (last=="infty")
		endevent.push_back(edm::EventID::maxEventNumber());
	      //		endevent.push_back(std::numeric_limits<long long int>::max());
	      else
		endevent.push_back((edm::EventNumber_t)atoi(last.c_str()));

	    }
	}
      listfile.close();
      // printout summary
      std::cout << "Summary from list of run/event number selection" << std::endl;
      for (unsigned int cond=0; cond<whattodo.size();cond++)
	{
	  std::string what;
	  if(whattodo[cond]) what="select";
	  else what="reject";
	  std::cout << what << " "; 
	  std::cout << "from run " << startrun[cond] << " to run " << endrun[cond] << " ";
	  std::cout << "from eve " << startevent[cond] << " to eve " << endevent[cond] << std::endl; 
	}
    }

  else std::cout << "Unable to open file"; 

}
void 
PickEvents::endJob() {
  using namespace std;
  std::cout << "================================================\n"
       << "  n Events Analyzed ............... " << nEventsAnalyzed << std::endl
       << "  n Events Selected ............... " << nEventsSelected<< std::endl
       << "================================================\n\n" ;
}


DEFINE_FWK_MODULE(PickEvents);
