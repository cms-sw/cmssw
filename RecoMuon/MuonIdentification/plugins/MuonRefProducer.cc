// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonRefProducer
// 
//
// Original Author:  Dmytro Kovalskyi
// $Id: MuonRefProducer.cc,v 1.5 2007/06/08 17:25:44 dmytro Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "RecoMuon/MuonIdentification/plugins/MuonRefProducer.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

MuonRefProducer::MuonRefProducer(const edm::ParameterSet& iConfig)
{
   theReferenceCollection = iConfig.getParameter<edm::InputTag>("ReferenceCollection");
   theSelector.setParameters(iConfig);
   produces<edm::RefVector<std::vector<reco::Muon> > >();
}


MuonRefProducer::~MuonRefProducer(){}

void MuonRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::auto_ptr<edm::RefVector<std::vector<reco::Muon> > > outputCollection(new edm::RefVector<std::vector<reco::Muon> >);

   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(theReferenceCollection, muons);
   
   // loop over input collection
   for ( unsigned int i=0; i<muons->size(); ++i ) 
     if ( theSelector.isGoodMuon( (*muons)[i] ) ) 
       outputCollection->push_back( edm::RefVector<std::vector<reco::Muon> >::value_type(muons,i) );
   iEvent.put(outputCollection);
}

bool MuonRefProducer::goodMuon( const reco::Muon& muon )
{
   return muon.pt() > 5;
}
