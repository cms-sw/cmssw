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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

#include <string>

//
// class declaration
//

class HLTCSCAcceptBusyFilter : public HLTFilter {

public:
  explicit HLTCSCAcceptBusyFilter(const edm::ParameterSet&);
  virtual ~HLTCSCAcceptBusyFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  bool AcceptManyHitsInChamber(unsigned int maxRecHitsPerChamber, const edm::Handle<CSCRecHit2DCollection>& recHits) const;

  // ----------member data ---------------------------
  edm::EDGetTokenT<CSCRecHit2DCollection> cscrechitsToken;
  edm::InputTag cscrechitsTag;
  bool          invert;
  unsigned int  maxRecHitsPerChamber;

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
HLTCSCAcceptBusyFilter::HLTCSCAcceptBusyFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
   //now do what ever initialization is needed
   cscrechitsTag        = iConfig.getParameter<edm::InputTag>("cscrechitsTag");
   invert               = iConfig.getParameter<bool>("invert");
   maxRecHitsPerChamber = iConfig.getParameter<unsigned int>("maxRecHitsPerChamber");
   cscrechitsToken      = consumes<CSCRecHit2DCollection>(cscrechitsTag);
}


HLTCSCAcceptBusyFilter::~HLTCSCAcceptBusyFilter()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


void
HLTCSCAcceptBusyFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("cscrechitsTag",edm::InputTag("hltCsc2DRecHits"));
  desc.add<bool>("invert",true);
  desc.add<unsigned int>("maxRecHitsPerChamber",200);
  descriptions.add("hltCSCAcceptBusyFilter",desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTCSCAcceptBusyFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

   using namespace edm;

  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> recHits;
  iEvent.getByToken(cscrechitsToken,recHits);

  if(  AcceptManyHitsInChamber(maxRecHitsPerChamber, recHits) ) {
    return (!invert);
  } else {
    return ( invert);
  }

}


// ------------ method to find chamber with nMax hits
bool HLTCSCAcceptBusyFilter::AcceptManyHitsInChamber(unsigned int maxRecHitsPerChamber, const edm::Handle<CSCRecHit2DCollection>& recHits) const {

  unsigned int maxNRecHitsPerChamber(0);

  const unsigned int nEndcaps(2);
  const unsigned int nStations(4);
  const unsigned int nRings(4);
  const unsigned int nChambers(36);
  unsigned int allRechits[nEndcaps][nStations][nRings][nChambers];
  for(unsigned int iE = 0;iE<nEndcaps;++iE){
    for(unsigned int iS = 0;iS<nStations;++iS){
      for(unsigned int iR = 0;iR<nRings;++iR){
	for(unsigned int iC = 0;iC<nChambers;++iC){
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

  for(unsigned int iE = 0;iE<nEndcaps;++iE){
    for(unsigned int iS = 0;iS<nStations;++iS){
      for(unsigned int iR = 0;iR<nRings;++iR){
	for(unsigned int iC = 0;iC<nChambers;++iC){
	  if(allRechits[iE][iS][iR][iC] > maxNRecHitsPerChamber) {
	    maxNRecHitsPerChamber = allRechits[iE][iS][iR][iC];
	  }
	  if(maxNRecHitsPerChamber > maxRecHitsPerChamber) {
	    return true;
	  }
	}
      }
    }
  }

  return false;

}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTCSCAcceptBusyFilter);
