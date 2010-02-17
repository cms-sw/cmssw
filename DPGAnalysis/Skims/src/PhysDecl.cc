// -*- C++ -*-
//
// Package:   BeamSplash
// Class:     BeamSPlash
//
//
// Original Author:  Luca Malgeri

#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "DPGAnalysis/Skims/interface/PhysDecl.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

using namespace edm;
using namespace std;

PhysDecl::PhysDecl(const edm::ParameterSet& iConfig)
{
  applyfilter = iConfig.getUntrackedParameter<bool>("applyfilter",true);
  debugOn     = iConfig.getUntrackedParameter<bool>("debugOn",false);
}

PhysDecl::~PhysDecl()
{
}

bool PhysDecl::filter( edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool accepted = false;
  // trigger info

  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  iEvent.getByLabel("gtDigis", gtrr_handle);
  L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();
 
  L1GtFdlWord fdlWord = gtrr->gtFdlWord();
  //   cout << "phys decl. bit=" << fdlWord.physicsDeclared() << endl;
  if (fdlWord.physicsDeclared() ==1) accepted=true;


  if (debugOn) {
    int ievt = iEvent.id().event();
    int irun = iEvent.id().run();
    int ils = iEvent.luminosityBlock();
    int bx = iEvent.bunchCrossing();
    
    cout << "PhysDecl_debug: Run " << irun << " Event " << ievt << " Lumi Block " << ils << " Bunch Crossing " << bx << " Accepted " << accepted << endl;
  }
 
  if (applyfilter)
    return accepted;
  else
    return true;

}

//define this as a plug-in
DEFINE_FWK_MODULE(PhysDecl);
