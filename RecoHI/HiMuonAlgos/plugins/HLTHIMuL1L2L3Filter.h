// -*- C++ -*-
//
// Package:    HLTHIMuL1L2L3.h
// Class:      HLTHIMuL1L2L3
/*/

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dong Ho Moon
//         Created:  Wed May  9 06:22:36 CEST 2007
// $Id: HLTHIMuL1L2L3Filter.h,v 1.3 2010/01/22 13:32:02 kodolova Exp $
//
//

#ifndef HLTHIMU_L1L2L3_FILTER_H
#define HLTHIMU_L1L2L3_FILTER_H


// system include files

#include <memory>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

// HI reco

#include "RecoHI/HiMuonAlgos/interface/HICConst.h"
#include "RecoHI/HiMuonAlgos/interface/FmpConst.h"
#include "RecoHI/HiMuonAlgos/interface/HITrackVertexMaker.h"

//
// class declaration
//
namespace cms{
class HLTHIMuL1L2L3Filter : public HLTFilter {

   private:
     edm::ParameterSet pset_;
    // HICConst * theHICConst;
    // FmpConst * theFmpConst;
    // HITrackVertexMaker * theTrackVertexMaker;

   public:

  //constructor

      explicit HLTHIMuL1L2L3Filter(const edm::ParameterSet&);
      ~HLTHIMuL1L2L3Filter();
      
  // General Block
  
      virtual bool hltFilter(edm ::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
      virtual void beginJob();
      virtual void endJob();
  
};
}
#endif
