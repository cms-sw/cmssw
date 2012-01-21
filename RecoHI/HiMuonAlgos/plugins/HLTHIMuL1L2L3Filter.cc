// -*- C++ -*-
//
// Package:    HLTHIMuL1L2L3
// Class:      HLTHIMuL1L2L3
//
// \class TestMuL1L2 TestMuL1L2.cc TestMuonL1L2/TestMuL1L2/src/TestMuL1L2.cc
//
// Original Author:  Dong Ho Moon
//         Created:  Wed May  9 06:22:36 CEST 2007
// $Id: HLTHIMuL1L2L3Filter.cc,v 1.2 2010/01/22 13:32:02 kodolova Exp $
//
//
// Comment: Dimuon reconstruction need primary vertex
//
 
#include "RecoHI/HiMuonAlgos/plugins/HLTHIMuL1L2L3Filter.h" 

#include <memory>

// C++ Headers

#include<iostream>
#include<iomanip>
#include<vector>
#include<cmath>

//  Framework 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace cms{
HLTHIMuL1L2L3Filter::HLTHIMuL1L2L3Filter(const edm::ParameterSet& ps1) : HLTFilter(ps1) 
{
   pset_ = ps1;
}

void HLTHIMuL1L2L3Filter::beginJob()
{
//   theHICConst = new HICConst();
//   theFmpConst = new FmpConst();
//   theTrackVertexMaker = new HITrackVertexMaker(pset_,es1);
}

void HLTHIMuL1L2L3Filter::endJob()
{
//   delete theHICConst;
//   delete theFmpConst;
//   delete theTrackVertexMaker;
}


//Destructor

HLTHIMuL1L2L3Filter::~HLTHIMuL1L2L3Filter()
{
} 

bool HLTHIMuL1L2L3Filter::hltFilter(edm::Event& e1, const edm::EventSetup& es1, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
   HITrackVertexMaker theTrackVertexMaker(pset_,es1);
// Start track finder
   HICConst theHICConst;
   FmpConst theFmpConst;
   bool dimuon = theTrackVertexMaker.produceTracks(e1,es1,&theHICConst,&theFmpConst);
//   if(dimuon) cout<<" The vertex is found : "<<endl; 
   return dimuon;
   
} 
} // namespace cms

