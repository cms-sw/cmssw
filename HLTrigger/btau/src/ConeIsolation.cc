// -*- C++ -*-
//
// Package:    ConeIsolation
// Class:      ConeIsolation
// 
/**\class ConeIsolation ConeIsolation.cc RecoBTag/ConeIsolation/src/ConeIsolation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Simone Gennai
//      Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: ConeIsolation.cc,v 1.2 2011/10/12 09:00:41 fwyzard Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/btau/src/ConeIsolation.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include <DataFormats/VertexReco/interface/Vertex.h>
#include <DataFormats/VertexReco/interface/VertexFwd.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

using namespace reco;
using namespace edm;
using namespace std;

//
// constructors and destructor
//
ConeIsolation::ConeIsolation(const edm::ParameterSet& iConfig)
{
  jetTrackSrc = iConfig.getParameter<InputTag>("JetTrackSrc");
  vertexSrc = iConfig.getParameter<InputTag>("vertexSrc");
  usingVertex = iConfig.getParameter<bool>("useVertex");
  usingBeamSpot = iConfig.getParameter<bool>("useBeamSpot"); //If false the OfflinePrimaryVertex will be used.
  beamSpotProducer = iConfig.getParameter<edm::InputTag>("BeamSpotProducer");
  m_algo = new ConeIsolationAlgorithm(iConfig);
  
  produces<reco::JetTagCollection>(); 
   produces<reco::IsolatedTauTagInfoCollection>();       



}


ConeIsolation::~ConeIsolation()
{
  delete m_algo;
}



//
// member functions
//
// ------------ method called to produce the data  ------------
void
ConeIsolation::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   //Get jets with tracks
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByLabel(jetTrackSrc,jetTracksAssociation);

   std::auto_ptr<reco::JetTagCollection>             tagCollection;
   std::auto_ptr<reco::IsolatedTauTagInfoCollection> extCollection( new reco::IsolatedTauTagInfoCollection() );
if (not jetTracksAssociation->empty()) {
     RefToBaseProd<reco::Jet> prod( jetTracksAssociation->begin()->first );
     tagCollection.reset( new reco::JetTagCollection(prod) );
   } else {
     tagCollection.reset( new reco::JetTagCollection() );
   }

   Vertex::Error e;
   e(0,0)=1;
   e(1,1)=1;
   e(2,2)=1;
   Vertex::Point p(0,0,-1000);
   Vertex myPVtmp(p,e);//Fake vertex to be used in case no vertex is found
   Vertex myPV;

   //Get pixel vertices
   Handle<reco::VertexCollection> vertices;
   iEvent.getByLabel(vertexSrc,vertices);
   const reco::VertexCollection vertCollection = *(vertices.product());
   //Check if there is the PV!!!!
   if(vertCollection.begin() != vertCollection.end())
     myPVtmp = *(vertCollection.begin());

   //In case the beam spot is used, the Z of the vertex still comes from the PV, while the (x,y) is taken from the beamspot
   reco::BeamSpot vertexBeamSpot;
   edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
   
   if(usingBeamSpot)
     {
       //Create a new vertex with the information on x0 and Y0 from the beamspot, to be used in HLT.
       iEvent.getByLabel(beamSpotProducer,recoBeamSpotHandle);
       vertexBeamSpot = *recoBeamSpotHandle;
       Vertex::Point bspoint(vertexBeamSpot.x0(),vertexBeamSpot.y0(),myPVtmp.z());
       Vertex combinedVertex = Vertex(bspoint,myPVtmp.error(),myPVtmp.chi2(),myPVtmp.ndof(),myPVtmp.tracksSize());
       myPV = combinedVertex;
	 }else{
	   myPV = myPVtmp;
	 }
   
   for (unsigned int i = 0; i < jetTracksAssociation->size(); ++i)
     {
       pair<float,IsolatedTauTagInfo> myPair =m_algo->tag(edm::Ref<JetTracksAssociationCollection>(jetTracksAssociation,i),myPV); 
       tagCollection->setValue(i, myPair.first);
       extCollection->push_back(myPair.second);
     }

   iEvent.put(extCollection);
   iEvent.put(tagCollection);

}

//define this as a plug-in
//DEFINE_FWK_MODULE(ConeIsolation);

