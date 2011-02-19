// -*- C++ -*-
//
// Package:    HiEcalSpikeFilter
// Class:      HiEcalSpikeFilter
// 
/**\class HiEcalSpikeFilter HiEcalSpikeFilter.cc RecoHI/HiEgammaAlgos/plugins/HiEcalSpikeFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yong Kim,32 4-A08,+41227673039,
//         Created:  Wed Oct 27 23:56:49 CEST 2010
// $Id: HiEcalSpikeFilter.cc,v 1.3 2010/10/28 13:26:07 kimy Exp $
//
//




// system include files
/*
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"




//
// class declaration
//

using namespace edm;
using namespace std;
using namespace reco;


class HiEcalSpikeFilter : public edm::EDFilter {
   public:
      explicit HiEcalSpikeFilter(const edm::ParameterSet&);
      ~HiEcalSpikeFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
   
   edm::InputTag photonSrc_;
   edm::InputTag ebReducedRecHitCollection_;
   edm::InputTag eeReducedRecHitCollection_;

   
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
HiEcalSpikeFilter::HiEcalSpikeFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   photonSrc_  =                      iConfig.getParameter<edm::InputTag>("photonProducer"); // photons
   ebReducedRecHitCollection_       = iConfig.getParameter<edm::InputTag>("ebReducedRecHitCollection"); //,"reducedEcalRecHitsEB");
   eeReducedRecHitCollection_       = iConfig.getParameter<edm::InputTag>("eeReducedRecHitCollection"); //,"reducedEcalRecHitsEE");
   
}


HiEcalSpikeFilter::~HiEcalSpikeFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HiEcalSpikeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   //using namespace edm;
   //using namespace std;
   //using namespace reco;

      
   //grab the photon collection
   Handle<reco::PhotonCollection> photonColl;
   iEvent.getByLabel(photonSrc_, photonColl);
   const reco::PhotonCollection *photons = photonColl.product();
   reco::PhotonCollection::const_iterator pho;
   reco::PhotonCollection::const_iterator leadingPho;

   //grab rechits
   edm::Handle<EcalRecHitCollection> EBReducedRecHits;
   iEvent.getByLabel(ebReducedRecHitCollection_, EBReducedRecHits);
   edm::Handle<EcalRecHitCollection> EEReducedRecHits;
   iEvent.getByLabel(eeReducedRecHitCollection_, EEReducedRecHits);
   //lazy tool
   EcalClusterLazyTools lazyTool(iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_ );
   // get the channel status from the DB                                                                                  
   edm::ESHandle<EcalChannelStatus> chStatus;
   iSetup.get<EcalChannelStatusRcd>().get(chStatus);
   
   

   // Find the leading gamma
   float leadingEt(-1), leadingEta(-1), leadingScEta(-1);
   
   if ( (*photons).size() == 0 )  // no photon candidates..
     return true;
   for (pho = (*photons).begin(); pho!= (*photons).end(); pho++){
     float tet       = (float)pho->et();
     if ( tet > leadingEt ) {
       leadingEt = tet;
	leadingPho = pho;
     }
   }
   
   leadingEt = leadingPho->et();
   leadingEta = leadingPho->eta();
   leadingScEta = leadingPho->superCluster()->eta();
   cout << " leading et = " << leadingEt << "GeV,   eta= " << leadingEta <<endl;
   
   // If the leading photon is at Endcap, then pass this event. 
   if ( fabs(leadingScEta) > 1.479)  return true;

   const reco::CaloClusterPtr  seed = leadingPho->superCluster()->seed();
   DetId id = lazyTool.getMaximum(*seed).first;
   const EcalRecHitCollection & rechits = ( leadingPho->isEB() ? *EBReducedRecHits : *EEReducedRecHits);
   EcalRecHitCollection::const_iterator it = rechits.find( id );
 
   int severity(-100), recoFlag(-100);
   
   if( it != rechits.end() ) {
      severity = EcalSeverityLevelAlgo::severityLevel( id, rechits, *chStatus );
      recoFlag = it->recoFlag();
   }
   cout << "severity= " << severity << "   recoFlag= " << recoFlag <<endl;
   //  cout << "definition of severity : http://cmslxr.fnal.gov/lxr/source/RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h?v=CMSSW_3_9_0_pre4" << endl;
   //  cout << "and that   of reco flag: http://cmslxr.fnal.gov/lxr/source/DataFormats/EcalRecHit/interface/EcalRecHit.h?v=CMSSW_3_9_0_pre4#074" << endl;
   
   bool finalFlag = true;
   if ( (severity==3) || (severity==4) || (recoFlag ==2) )  
     finalFlag = false;
   else 
     finalFlag = true;
   
return finalFlag;
}



// ------------ method called once each job just before starting event loop  ------------
void 
HiEcalSpikeFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiEcalSpikeFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiEcalSpikeFilter);
