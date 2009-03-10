// -*- C++ -*-
//
// Package:    TestMuL1L2.h
// Class:      TestMuL1L2
/*/

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dong Ho Moon
//         Created:  Wed May  9 06:22:36 CEST 2007
// $Id: TestMuL1L2Filter.h,v 1.2 2008/09/14 12:25:19 kodolova Exp $
//
//

#ifndef TESTMU_L1L2_FILTER_H
#define TESTMU_L1L2_FILTER_H


// system include files

#include <memory>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"

// HI reco

#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "RecoHIMuon/HiMuPropagator/interface/FmpConst.h"
#include "RecoHIMuon/HiMuTracking/interface/HITrackVertexMaker.h"

//
// class declaration
//
namespace cms{
class TestMuL1L2Filter : public edm::EDFilter {

   private:
     edm::ParameterSet pset_;
     HICConst * theHICConst;
     FmpConst * theFmpConst;
     HITrackVertexMaker * theTrackVertexMaker;

   public:

  //constructor

      explicit TestMuL1L2Filter(const edm::ParameterSet&);
      ~TestMuL1L2Filter();
      
  // General Block
  
      virtual bool filter(edm ::Event&, const edm::EventSetup&);
      virtual void beginJob(const edm::EventSetup& es1);
      virtual void endJob();
  
};
}
#endif
