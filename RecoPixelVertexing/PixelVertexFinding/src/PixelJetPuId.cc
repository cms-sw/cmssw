// -*- C++ -*-
//
// Package:    PixelJetPuId
// Class:      PixelJetPuId
// 
/**\class PixelJetPuId PixelJetPuId.cc RecoBTag/PixelJetPuId/src/PixelJetPuId.cc

 Description: The PixelJetPuId module select all the pixel tracks compatible with a jet. If the sum of the tracks momentum is under a threshold the jet is tagged as "PUjets".

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Silvio DONATO
//         Created:  Wed Dec 18 10:05:40 CET 2013
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

//
// class declaration
//

class PixelJetPuId : public edm::EDFilter {
   public:
      explicit PixelJetPuId(const edm::ParameterSet&);
      ~PixelJetPuId();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual bool beginRun(edm::Run&, edm::EventSetup const&);
      virtual bool endRun(edm::Run&, edm::EventSetup const&);
      virtual bool beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual bool endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
     edm::InputTag m_associator; 
     edm::InputTag m_primaryVertex;
     edm::InputTag m_beamSpot;
     
     double m_MinTrackPt; 
     double m_MaxTrackChi2; 
     double m_MaxTrackDistanceToJet; 

     double m_MinGoodJetTrackPt;
     double m_MinGoodJetTrackPtRatio; 
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
PixelJetPuId::PixelJetPuId(const edm::ParameterSet& iConfig)
{
  //InputTag
  m_beamSpot          = iConfig.getParameter<edm::InputTag>("beamSpot");
  m_associator        = iConfig.getParameter<edm::InputTag>("jetTracks");
  m_primaryVertex              = iConfig.getParameter<edm::InputTag>("primaryVertex");

  //Tracks Selection
  m_MinTrackPt     = iConfig.getParameter<double>("MinTrackPt");
  m_MaxTrackDistanceToJet = iConfig.getParameter<double>("MaxTrackDistanceToJet");
  m_MaxTrackChi2 = iConfig.getParameter<double>("MaxTrackChi2");

  //A jet is defined as a signal jet if Sum(trackPt) > minPt or Sum(comp.trackPt)/CaloJetPt > minPtRatio
  m_MinGoodJetTrackPt          = iConfig.getParameter<double>("MinGoodJetTrackPt");
  m_MinGoodJetTrackPtRatio     = iConfig.getParameter<double>("MinGoodJetTrackPtRatio");
 
  produces<std::vector<reco::CaloJet> >(); 
  produces<std::vector<reco::CaloJet> >("PUjets"); 
}


PixelJetPuId::~PixelJetPuId()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
PixelJetPuId::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   std::auto_ptr<std::vector<reco::CaloJet> > pOut(new std::vector<reco::CaloJet> );
   std::auto_ptr<std::vector<reco::CaloJet> > pOut_PUjets(new std::vector<reco::CaloJet> );

   //get jetTracksAssociation
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByLabel(m_associator, jetTracksAssociation);
   
   //get primary vertices
   Handle<reco::VertexCollection> primaryVertex;
   iEvent.getByLabel(m_primaryVertex, primaryVertex);

   //get Transient Track Builder
   edm::ESHandle<TransientTrackBuilder> builder;
   iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);

   reco::JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();

   //loop on trackIPTagInfos
   if(primaryVertex->size()>0)
   {
   	const reco::Vertex* pv = &*primaryVertex->begin();
   	//loop on jets
	for( ;  it != jetTracksAssociation->end(); it++ ) {
		const reco::TrackRefVector& tracks = it->second;

		math::XYZVector jetMomentum = it->first->momentum();
	        GlobalVector direction(jetMomentum.x(), jetMomentum.y(), jetMomentum.z());

		int ntracks=0;
		math::XYZVector trMomentum; 

		reco::TrackRefVector::const_iterator itTrack = tracks.begin();      
		//loop on tracks
		for(; itTrack != tracks.end(); ++itTrack) 
		{
			reco::TransientTrack transientTrack = builder->build(*itTrack);
	     		float jetTrackDistance = ((IPTools::jetTrackDistance(transientTrack, direction, *pv)).second).value();
	     		
			//select the tracks compabible with the jet
			if(( (*itTrack)->pt() > m_MinTrackPt) && ( (*itTrack)->normalizedChi2() < m_MaxTrackChi2) && (jetTrackDistance>-m_MaxTrackDistanceToJet))
			{
				ntracks++;
				trMomentum += (*itTrack)->momentum(); //calculate the Sum(trackPt)
			}
		}
		//if Sum(comp.trackPt)/CaloJetPt > minPtRatio or Sum(trackPt) > minPt  the jet is a signal jet
		if(trMomentum.rho()/jetMomentum.rho() > m_MinGoodJetTrackPtRatio || trMomentum.rho() > m_MinGoodJetTrackPt ) 
		{
			pOut->push_back(* dynamic_cast<const reco::CaloJet *>(&(*it->first)));// fill it as signal jet
		}
		else//else it is a PUjet
		{
			pOut_PUjets->push_back(* dynamic_cast<const reco::CaloJet *>(&(*it->first)));// fill it as PUjets
		}
	 }
   }
   iEvent.put(pOut);
   iEvent.put(pOut_PUjets,"PUjets");

   edm::Handle<reco::BeamSpot> beamSpot;
   iEvent.getByLabel(m_beamSpot,beamSpot);
 
   return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
PixelJetPuId::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PixelJetPuId::endJob() {
}

// ------------ method called when starting to processes a run  ------------
bool 
PixelJetPuId::beginRun(edm::Run&, edm::EventSetup const&)
{ 
  return true;
}

// ------------ method called when ending the processing of a run  ------------
bool 
PixelJetPuId::endRun(edm::Run&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when starting to processes a luminosity block  ------------
bool 
PixelJetPuId::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when ending the processing of a luminosity block  ------------
bool 
PixelJetPuId::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PixelJetPuId::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PixelJetPuId);
