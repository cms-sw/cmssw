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
// $Id: InvariantMass.cc,v 1.10 2007/10/07 13:01:05 fwyzard Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <utility>
#include <boost/regex.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "RecoTauTag/InvariantMass/interface/InvariantMass.h"

using namespace std;

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
   
   std::auto_ptr<JetTagCollection>         tagCollection;
   std::auto_ptr<TauMassTagInfoCollection> extCollection( new TauMassTagInfoCollection() );

   // Island basic cluster collection
   Handle<BasicClusterCollection> barrelBasicClusterHandle;
   iEvent.getByLabel(m_ecalBClSrc, "islandBarrelBasicClusters", barrelBasicClusterHandle);
   const reco::BasicClusterCollection & barrelClusters = *(barrelBasicClusterHandle.product());

   Handle<BasicClusterCollection> endcapBasicClusterHandle;
   iEvent.getByLabel(m_ecalBClSrc, "islandEndcapBasicClusters", endcapBasicClusterHandle);
   const reco::BasicClusterCollection & endcapClusters = *(endcapBasicClusterHandle.product());

   if (isolatedTaus->empty()) {
     tagCollection.reset( new JetTagCollection() );
   } else {
     RefToBaseProd<reco::Jet> prod( isolatedTaus->begin()->jet() );
     tagCollection.reset( new JetTagCollection(RefToBaseProd<reco::Jet>(prod)) );

     for (unsigned int i = 0; i < isolatedTaus->size(); ++i)
     {
       IsolatedTauTagInfoRef tauRef(isolatedTaus, i);
       const Jet & jet = *(tauRef->jet()); 
       math::XYZVector jetDir(jet.px(),jet.py(),jet.pz());  
       pair<double,TauMassTagInfo> jetTauPair;
       if (jetDir.eta() < 1.2)      // barrel  
         jetTauPair = m_algo->tag(iEvent, iSetup, tauRef, barrelBasicClusterHandle);
       else                         // endcap
         jetTauPair = m_algo->tag(iEvent, iSetup, tauRef, endcapBasicClusterHandle);
       tagCollection->setValue( i, jetTauPair.first );
       extCollection->push_back( jetTauPair.second );
     }
   }

   iEvent.put( tagCollection );
   iEvent.put( extCollection );
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(InvariantMass);
