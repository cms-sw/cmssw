// -*- C++ -*-
//
// Package:    HLTCSCAcceptBusyFilter
// Class:      HLTCSCAcceptBusyFilter
// 
/**\class HLTCSCAcceptBusyFilter HLTCSCAcceptBusyFilter.cc Analyzers/HLTCSCAcceptBusyFilter/src/HLTCSCAcceptBusyFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ingo Bloch
//         Created:  Mon Mar 15 11:39:08 CDT 2010
// $Id: HLTCSCAcceptBusyFilter.cc,v 1.1 2010/03/17 22:06:22 stoyan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
//include "FWCore/Framework/interface/EDFilter.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>

#include <string>

//
// class declaration
//

class HLTCSCAcceptBusyFilter : public HLTFilter {
   public:
      explicit HLTCSCAcceptBusyFilter(const edm::ParameterSet&);
      virtual ~HLTCSCAcceptBusyFilter();

   private:
  //      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
  //      virtual void endJob() ;
  
  bool AcceptManyHitsInChamber(uint maxRecHitsPerChamber, edm::Handle<CSCRecHit2DCollection> recHits);
  //, edm::ESHandle<CSCGeometry> cscGeom);
  
  edm::InputTag cscrechitsTag;
  bool          invertResult;
  uint          maxRecHitsPerChamber;
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HLTCSCAcceptBusyFilter::HLTCSCAcceptBusyFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   cscrechitsTag         = iConfig.getParameter<edm::InputTag>("cscrechitsTag");
   invertResult          = iConfig.getParameter<bool>("invertResult");
   maxRecHitsPerChamber  = iConfig.getParameter<uint>("maxRecHitsPerChamber");

}


HLTCSCAcceptBusyFilter::~HLTCSCAcceptBusyFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTCSCAcceptBusyFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  // Get the RecHits collection :
  edm::Handle<CSCRecHit2DCollection> recHits; 
  
  
  iEvent.getByLabel(cscrechitsTag,recHits);  
  
  if(  AcceptManyHitsInChamber(maxRecHitsPerChamber, recHits) ) {
    return (!invertResult);
  } else {
    return ( invertResult);
  }

}


// ------------ method to find chamber with nMax hits
bool HLTCSCAcceptBusyFilter::AcceptManyHitsInChamber(uint maxRecHitsPerChamber, edm::Handle<CSCRecHit2DCollection> recHits) {
  uint maxNRecHitsPerChamber = 0;
  bool takeEvent = false;
  const uint nEndcaps = 2;
  const uint nStations = 4;
  const uint nRings = 4;
  const uint nChambers = 36;
  uint allRechits[nEndcaps][nStations][nRings][nChambers];
  for(uint iE = 0;iE<nEndcaps;++iE){
    for(uint iS = 0;iS<nStations;++iS){
      for(uint iR = 0;iR<nRings;++iR){
	for(uint iC = 0;iC<nChambers;++iC){
	  allRechits[iE][iS][iR][iC] = 0;
	}
      }
    }
  }
  for(CSCRecHit2DCollection::const_iterator it = recHits->begin(); it != recHits->end(); it++) {

    ++allRechits
      [(*it).cscDetId().endcap()-1]
      [(*it).cscDetId().station()-1]
      [(*it).cscDetId().ring()-1]
      [(*it).cscDetId().chamber()-1];
  }

  for(uint iE = 0;iE<nEndcaps;++iE){
    for(uint iS = 0;iS<nStations;++iS){
      for(uint iR = 0;iR<nRings;++iR){
	for(uint iC = 0;iC<nChambers;++iC){
	  if(allRechits[iE][iS][iR][iC] > maxNRecHitsPerChamber) {
	    maxNRecHitsPerChamber = allRechits[iE][iS][iR][iC];
	  }
	  if(maxNRecHitsPerChamber > maxRecHitsPerChamber) {
	    takeEvent = true;
	    return takeEvent;
	  }
	}
      }
    }
  }
  return takeEvent;
}


// // ------------ method called once each job just before starting event loop  ------------
// void 
// HLTCSCAcceptBusyFilter::beginJob()
// {
// }

// // ------------ method called once each job just after ending the event loop  ------------
// void 
// HLTCSCAcceptBusyFilter::endJob() {
// }

//define this as a plug-in
DEFINE_FWK_MODULE(HLTCSCAcceptBusyFilter);
