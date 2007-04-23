// -*- C++ -*-
//
// $Id: HltAnalyzer.cc,v 1.6 2007/03/10 10:31:26 dlange Exp $
//
/**\class HltAnalyzer HltAnalyzer.cc DQM/HLTEvF/src/HltAnalyzer.cc

   Description: 
      Analyze data from an HLTPerformanceInfo object

   Implementation:
     - generate some histograms in the event loop
     - also fill some maps to be used in the endJob - this 
       needs to be rethought for DQM

**/
//
// Original Author:  Peter Wittich
//         Created:  Thu Nov  9 07:51:28 CST 2006
// $Id: HltAnalyzer.cc,v 1.6 2007/03/10 10:31:26 dlange Exp $
//
//
//

#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/HLTEvF/interface/HltAnalyzer.h"

#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"

#include "TFile.h"
#include "TH1D.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HltAnalyzer::HltAnalyzer(const edm::ParameterSet& iConfig)
  : myName_(iConfig.getParameter<std::string>("@module_label")),
    verbose_(iConfig.getUntrackedParameter("verbose",false)),
    hltPerfLabel_(iConfig.getParameter<edm::InputTag>("hltPerfLabel")),
    slowestModule_(),
    rejectionModule_(),
    f_(0),
    s1_(0),
    s2_(0)
{

}


HltAnalyzer::~HltAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HltAnalyzer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //
  edm::Handle<HLTPerformanceInfo> hperf;
  using namespace edm;
  try {
    iEvent.getByLabel(hltPerfLabel_, hperf);
  } 
  catch(...) { // and drop...
    ;
  }



  if ( ! hperf.isValid() ) {
    std::cout << name() << ": did not find HLTPerformanceInfo with label " 
	      << hltPerfLabel_
	      << std::endl;
    return false;
  }

  // for each path, find the module that caused the reject
  for ( HLTPerformanceInfo::PathList::const_iterator i = hperf->beginPaths();
 	i != hperf->endPaths(); ++i ) {
     ++rejectionModule_[i->name()][i->lastModuleByStatusName()];
  }
  std::string slowpoke("nobody");
  float slowtime = -99;
  
  for ( HLTPerformanceInfo::Modules::const_iterator m = hperf->beginModules();
	m != hperf->endModules(); ++m ) {
    if ( m->time() > slowtime ) {
      slowpoke = m->name(); slowtime = m->time();
    }
  }
  ++slowestModule_[slowpoke];
  s1_->Fill(slowtime);
  s2_->Fill(hperf->totalTime());



  return true;
}

// ------- method called once each job just before starting event loop  -----
void 
HltAnalyzer::beginJob(const edm::EventSetup&)
{
  f_ = TFile::Open("histos.root", "RECREATE");
  // unit is seconds
  s1_ = new TH1D("s1", "slowest module times", 100, 0., .200);
  
}

// ------ method called once each job just after ending the event loop  ----
void 
HltAnalyzer::endJob() {
  std::cout << name() << ": making final histograms." << std::endl;
  if ( f_->IsZombie() ) {
    std::cout << name() << ": opening file failed?" << std::endl;
    return ;
  }
  f_->cd();
  int nmodules = slowestModule_.size();
  TH1D *s0 = new TH1D("s0", "slowest module count", nmodules, 0, nmodules);
  int cnt = 1;
  ModuleCount_t::const_iterator i = slowestModule_.begin();
  for ( ; i != slowestModule_.end(); ++i ) {
    s0->GetXaxis()->SetBinLabel(cnt, i->first.c_str());
    s0->SetBinContent(cnt, i->second);
    std::cout << i->first << ", " << i->second << std::endl;
    ++cnt;
  }
  int npath = rejectionModule_.size();
  TH1D **p = new TH1D*[npath];
  int np = 0;
  PathModuleCount_t::const_iterator j = rejectionModule_.begin();
  for ( ; j != rejectionModule_.end(); ++j ) {
    char name[128], title[128];
    snprintf(name, 128, "p%d", np);
    snprintf(title, 128, "Path %s", j->first.c_str());
    int nmod = j->second.size();
    p[np]= new TH1D(name, title, nmod, 0, nmod);
    i = j->second.begin();
    int cnt = 1;
    for ( ; i != j->second.end(); ++i ) {
      p[np]->GetXaxis()->SetBinLabel(cnt, i->first.c_str());
      p[np]->SetBinContent(cnt, i->second);
      ++cnt;
    }
    ++np;
  }
  f_->Write();
  f_->Close();
  // leak memory here cuz I don't understand how root considers ownership.
  return;
}


