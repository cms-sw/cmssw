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
// $Id: TestMuL1L2FilterSTA.h,v 1.2 2010/01/22 13:32:02 kodolova Exp $
//
//

#ifndef TESTMU_L1L2_FILTERSTA_H
#define TESTMU_L1L2_FILTERSTA_H


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

#include "RecoHI/HiMuonAlgos/interface/HICConst.h"
#include "RecoHI/HiMuonAlgos/interface/FmpConst.h"
#include "RecoHI/HiMuonAlgos/interface/HITrackVertexMaker.h"

//
// class declaration
//
namespace cms{
class TestMuL1L2FilterSTA : public edm::EDFilter {

   private:
     edm::ParameterSet pset_;
     //HICConst * theHICConst;
     //FmpConst * theFmpConst;
     //HITrackVertexMaker * theTrackVertexMaker;

   public:

  //constructor

      explicit TestMuL1L2FilterSTA(const edm::ParameterSet&);
      ~TestMuL1L2FilterSTA();
      
  // General Block
  
      virtual bool filter(edm ::Event&, const edm::EventSetup&);
      virtual void beginJob();
      virtual void endJob();
  
};
}
#endif
