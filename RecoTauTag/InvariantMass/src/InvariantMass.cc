// -*- C++ -*-
//
// Package:    InvariantMass
// Class:      InvariantMass
// 
/**\class InvariantMass InvariantMass.cc RecoBTag/InvariantMass/src/InvariantMass.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Suchandra Dutta
//      Created:  Thu Oct 19 09:02:32 CEST 2006
// $Id: InvariantMass.cc,v 1.0 2006/10/9 09:02:32 dutta Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoTauTag/InvariantMass/interface/InvariantMass.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include <DataFormats/VertexReco/interface/Vertex.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include <boost/regex.hpp>


//
// constructors and destructor
//
InvariantMass::InvariantMass(const edm::ParameterSet& iConfig)
{
  jetTrackSrc = iConfig.getParameter<string>("JetTrackSrc");
  m_ecalBClSrc = iConfig.getParameter<string>("ecalbcl");

  m_algo = new InvariantMassAlgorithm(iConfig);
  
  produces<reco::JetTagCollection>();  //Several producer so I put a label
  produces<reco::TauMassTagInfoCollection>();   //Only one producer


}


InvariantMass::~InvariantMass()
{
  delete m_algo;
}



//
// member functions
//
// ------------ method called to produce the data  ------------
void
InvariantMass::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   Handle<IsolatedTauTagInfoCollection> isolatedTaus;
   iEvent.getByLabel(jetTrackSrc, isolatedTaus);
   
   JetTagCollection         *baseCollection = new JetTagCollection();
   TauMassTagInfoCollection *extCollection  = new TauMassTagInfoCollection();


   // Island basic cluster collection
   Handle<BasicClusterCollection> barrelBasicClusterHandle;
   iEvent.getByLabel(m_ecalBClSrc,"islandBarrelBasicClusters",barrelBasicClusterHandle);
   reco::BasicClusterCollection barrelClusters = *(barrelBasicClusterHandle.product());

   Handle<BasicClusterCollection> endcapBasicClusterHandle;
   iEvent.getByLabel(m_ecalBClSrc,"islandEndcapBasicClusters",endcapBasicClusterHandle);
   reco::BasicClusterCollection endcapClusters = *(endcapBasicClusterHandle.product());

   reco::BasicClusterCollection allClusters;
   //   reco::BasicClusterRef clusterRef;
   for (BasicClusterCollection::const_iterator iclus1 = barrelClusters.begin();
	iclus1 != barrelClusters.end(); iclus1++) allClusters.push_back(*iclus1);
   for (BasicClusterCollection::const_iterator iclus2 = endcapClusters.begin();
	iclus2 != endcapClusters.end(); iclus2++) allClusters.push_back(*iclus2);


   int theKey = 0;
   for(IsolatedTauTagInfoCollection::const_iterator it = isolatedTaus->begin(); 
                 it != isolatedTaus->end(); it++) {
     IsolatedTauTagInfoRef tauRef(isolatedTaus,theKey);
     math::XYZVector jetDir(tauRef->jet().px(),tauRef->jet().py(),tauRef->jet().pz());
     pair<JetTag,TauMassTagInfo> jetTauPair;
     if (jetDir.eta() < 1.2) { //barrel  
       jetTauPair = m_algo->tag(iEvent,iSetup,tauRef,barrelBasicClusterHandle);
     } else {
       jetTauPair = m_algo->tag(iEvent,iSetup,tauRef, endcapBasicClusterHandle);
     }
     baseCollection->push_back(jetTauPair.first);    
     extCollection->push_back(jetTauPair.second);
     theKey++;
   }

   std::auto_ptr<reco::JetTagCollection> resultBase(baseCollection);
   edm::OrphanHandle <reco::JetTagCollection >  myJetTag =  iEvent.put(resultBase);
   
   int ic=0;
   for (reco::TauMassTagInfoCollection::iterator im = extCollection->begin(); 
               im != extCollection->end(); im++) {
     im->setJetTag(JetTagRef(myJetTag,ic)); 
     ic++;
   }
   
   std::auto_ptr<reco::TauMassTagInfoCollection> resultExt(extCollection);  
   iEvent.put(resultExt);
}


