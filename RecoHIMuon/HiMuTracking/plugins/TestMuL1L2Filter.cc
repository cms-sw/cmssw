// -*- C++ -*-
//
// Package:    TestMuL1L2
// Class:      TestMuL1L2
//
// \class TestMuL1L2 TestMuL1L2.cc TestMuonL1L2/TestMuL1L2/src/TestMuL1L2.cc
//
// Original Author:  Dong Ho Moon
//         Created:  Wed May  9 06:22:36 CEST 2007
// $Id: TestMuL1L2Filter.cc,v 1.5 2009/02/11 15:37:04 kodolova Exp $
//
//
// Comment: Dimuon reconstruction need primary vertex
//
 
#include "RecoHIMuon/HiMuTracking/plugins/TestMuL1L2Filter.h" 

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
TestMuL1L2Filter::TestMuL1L2Filter(const edm::ParameterSet& ps1)
{
   pset_ = ps1;
}

void TestMuL1L2Filter::beginJob(const edm::EventSetup& es1)
{
   theHICConst = new HICConst();
   theFmpConst = new FmpConst();
   theTrackVertexMaker = new HITrackVertexMaker(pset_,es1);
}

void TestMuL1L2Filter::endJob()
{
   delete theHICConst;
   delete theFmpConst;
   delete theTrackVertexMaker;
   
}


//Destructor

TestMuL1L2Filter::~TestMuL1L2Filter()
{
} 

bool TestMuL1L2Filter::filter(edm::Event& e1, const edm::EventSetup& es1)
{

// Start track finder

   bool dimuon = theTrackVertexMaker->produceTracks(e1,es1,theHICConst,theFmpConst);
//   if(dimuon) cout<<" The vertex is found : "<<endl; 
   return dimuon;
   
} 
} // namespace cms

