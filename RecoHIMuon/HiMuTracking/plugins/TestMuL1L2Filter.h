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
// $Id: TestMuL1L2Filter.h,v 1.1 2008/07/14 16:59:27 kodolova Exp $
//
//

#ifndef TESTMU_L1L2_FILTER_H
#define TESTMU_L1L2_FILTER_H


// system include files

#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoHIMuon/HiMuSeed/interface/HICConst.h"
#include "RecoHIMuon/HiMuPropagator/interface/FmpConst.h"
#include "RecoHIMuon/HiMuTracking/interface/HITrackVertexMaker.h"

//-------------------------------------
// L1 Trigger header files
//-------------------------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
//-------------------------------------
// L2 Trigger Header files
//-------------------------------------
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//
// class declaration
//
namespace cms{
class TestMuL1L2Filter : public edm::EDFilter {

   private:

  edm::InputTag m_inTag;
  edm::InputTag candTag_; 
  edm::InputTag rphirecHitsTag;
  std::string recHitBuilderName;
   public:

  //constructor

      explicit TestMuL1L2Filter(const edm::ParameterSet&);
      ~TestMuL1L2Filter();
      virtual bool filter(edm ::Event&, const edm::EventSetup&);
      virtual void beginJob(const edm::EventSetup& es1);
      virtual void endJob();

  // General Block
  int runno;
  int eveno;
  TrackerGeometry* trackerg;
  edm::ParameterSet pset_;
  HICConst * theHICConst;
  FmpConst * theFmpConst;
  HITrackVertexMaker * theTrackVertexMaker;
};
}
#endif
