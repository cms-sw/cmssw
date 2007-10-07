// -*- C++ -*-
//
// Package:    CaloMuonProducer
// Class:      CaloMuonProducer
// 
/**\class CaloMuonProducer CaloMuonProducer.cc Test/CaloMuonProducer/src/CaloMuonProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Wed Oct  3 16:29:03 CDT 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "RecoMuon/MuonIdentification/interface/MuonCaloCompatibility.h"


class CaloMuonProducer : public edm::EDProducer {
 public:
   explicit CaloMuonProducer(const edm::ParameterSet&);
   ~CaloMuonProducer();
   
 private:
   virtual void     produce( edm::Event&, const edm::EventSetup& );
   reco::CaloMuon   makeMuon( const edm::Event& iEvent, 
			      const edm::EventSetup& iSetup,
			      const reco::TrackRef& track );

   double caloCut_;
   TrackDetectorAssociator trackAssociator_;
   TrackAssociatorParameters parameters_;
   MuonCaloCompatibility muonCaloCompatibility_;
   edm::InputTag inputMuons_;
   edm::InputTag inputTracks_;
};
