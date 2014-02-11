// -*- C++ -*-
//
// Package:    PixelJetPuId
// Class:      PixelJetPuId
// 
/**\class PixelJetPuId PixelJetPuId.cc RecoBTag/PixelJetPuId/src/PixelJetPuId.cc

 Description:
 The PixelJetPuId module select all the pixel tracks compatible with a jet.
 If the sum of the tracks momentum is under a threshold the jet is tagged as "PUjets".

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Silvio DONATO
//         Created:  Wed Dec 18 10:05:40 CET 2013
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/Math/interface/deltaR.h"
     
//
// class declaration
//

class PixelJetPuId : public edm::EDProducer {
   public:
      PixelJetPuId(const edm::ParameterSet&);
      virtual ~PixelJetPuId();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      

      // ----------member data ---------------------------
     edm::InputTag m_primaryVertex;
     edm::InputTag m_tracks;
     edm::InputTag m_jets;
     edm::EDGetTokenT<std::vector<reco::Track> > tracksToken;
     edm::EDGetTokenT<edm::View<reco::CaloJet> > jetsToken;
     edm::EDGetTokenT<reco::VertexCollection> primaryVertexToken;

     double m_MinTrackPt; 
     double m_MaxTrackChi2; 
     double m_MaxTrackDistanceToJet; 

     bool   m_fwjets;
     double m_mineta_fwjets;
     double m_minet_fwjets;

     double m_MinGoodJetTrackPt;
     double m_MinGoodJetTrackPtRatio; 
};


//
// constructors and destructor
//
PixelJetPuId::PixelJetPuId(const edm::ParameterSet& iConfig)
{
  //InputTag
  m_tracks           = iConfig.getParameter<edm::InputTag>("tracks");
  tracksToken        = consumes<std::vector<reco::Track> >(m_tracks);
  m_jets             = iConfig.getParameter<edm::InputTag>("jets");
  jetsToken          = consumes<edm::View<reco::CaloJet> >(m_jets);
  m_primaryVertex    = iConfig.getParameter<edm::InputTag>("primaryVertex");
  primaryVertexToken = consumes<reco::VertexCollection>(m_primaryVertex);

  //Tracks Selection
  m_MinTrackPt             = iConfig.getParameter<double>("MinTrackPt");
  m_MaxTrackDistanceToJet  = iConfig.getParameter<double>("MaxTrackDistanceToJet");
  m_MaxTrackChi2           = iConfig.getParameter<double>("MaxTrackChi2");

  //A jet is defined as a signal jet if Sum(trackPt) > minPt or Sum(comp.trackPt)/CaloJetPt > minPtRatio
  m_MinGoodJetTrackPt      = iConfig.getParameter<double>("MinGoodJetTrackPt");
  m_MinGoodJetTrackPtRatio = iConfig.getParameter<double>("MinGoodJetTrackPtRatio");

  m_fwjets                 = iConfig.getParameter<bool>("UseForwardJetsAsNoPU");
  m_mineta_fwjets          = iConfig.getParameter<double>("MinEtaForwardJets");
  m_minet_fwjets           = iConfig.getParameter<double>("MinEtForwardJets");

  produces<std::vector<reco::CaloJet> >(); 
  produces<std::vector<reco::CaloJet> >("PUjets"); 
}


PixelJetPuId::~PixelJetPuId() {}


void
PixelJetPuId::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag> ("jets",edm::InputTag("hltCaloJetL1FastJetCorrected"));
  desc.add<edm::InputTag> ("tracks",edm::InputTag("hltPixelTracksNoPU"));
  desc.add<edm::InputTag> ("primaryVertex",edm::InputTag("hltFastPVPixelVertices"));
  desc.add<double>("MinGoodJetTrackPtRatio",0.045);
  desc.add<double>("MinGoodJetTrackPt",1.8);
  desc.add<double>("MaxTrackDistanceToJet",0.04);
  desc.add<double>("MinTrackPt",0.6);
  desc.add<double>("MaxTrackChi2",20.);
  desc.add<bool>("UseForwardJetsAsNoPU",true);
  desc.add<double>("MinEtaForwardJets",2.4);
  desc.add<double>("MinEtForwardJets",40.);
  descriptions.add("pixelJetPuId",desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
void PixelJetPuId::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  std::auto_ptr<std::vector<reco::CaloJet> > pOut(new std::vector<reco::CaloJet> );
  std::auto_ptr<std::vector<reco::CaloJet> > pOut_PUjets(new std::vector<reco::CaloJet> );
  
   //get tracks
  Handle<std::vector<reco::Track> > tracks;
  iEvent.getByToken(tracksToken, tracks);
  unsigned int tsize = tracks->size();
  float teta[tsize], tphi[tsize];
  unsigned int i=0;
  for (auto const & tr : *tracks) { teta[i]=tr.eta(); tphi[i]=tr.phi();++i;}
   
  //get jets
  Handle<edm::View<reco::CaloJet> > jets;
  iEvent.getByToken(jetsToken, jets);
   
  //get primary vertices
  Handle<reco::VertexCollection> primaryVertex;
  iEvent.getByToken(primaryVertexToken, primaryVertex);

  //get Transient Track Builder
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);

  //loop on trackIPTagInfos
  if(primaryVertex->size()>0)
    {
      const reco::Vertex* pv = &*primaryVertex->begin();
      //loop on jets
      for(edm::View<reco::CaloJet>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++ ) {
	
	math::XYZVector jetMomentum = itJet->momentum();
	GlobalVector direction(jetMomentum.x(), jetMomentum.y(), jetMomentum.z());
	
	math::XYZVector trMomentum;
	
	//loop on tracks
	if(fabs(itJet->eta())>m_mineta_fwjets)
	  {
	    if((m_fwjets) && (itJet->et()>m_minet_fwjets))
	      pOut->push_back(*itJet);// fill forward jet as signal jet
	  }
	else 
	  {
	    std::vector<reco::Track>::const_iterator itTrack = tracks->begin();
	    for (unsigned int i=0; i<tsize; ++i) {
	      float deltaR2=reco::deltaR2(itJet->eta(),itJet->phi(), teta[i],tphi[i]);
	      if(deltaR2<0.25) {
		reco::TransientTrack transientTrack = builder->build(*itTrack);
		float jetTrackDistance = -((IPTools::jetTrackDistance(transientTrack, direction, *pv)).second).value();
		    
		//select the tracks compabible with the jet
		if(( itTrack->pt() > m_MinTrackPt) && ( itTrack->normalizedChi2() < m_MaxTrackChi2) && (jetTrackDistance<m_MaxTrackDistanceToJet))
		  {
		    trMomentum += itTrack->momentum(); //calculate the Sum(trackPt)
		  }
	      }
	      itTrack++;
	    }
	    //if Sum(comp.trackPt)/CaloJetPt > minPtRatio or Sum(trackPt) > minPt  the jet is a signal jet
	    if(trMomentum.rho()/jetMomentum.rho() > m_MinGoodJetTrackPtRatio || trMomentum.rho() > m_MinGoodJetTrackPt ) 
	      {
		pOut->push_back(*itJet);        // fill it as signal jet
	      }
	    else//else it is a PUjet
	      {
		pOut_PUjets->push_back(*itJet); // fill it as PUjets
	      }
	  }
      }
    }
  iEvent.put(pOut);
  iEvent.put(pOut_PUjets,"PUjets");
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelJetPuId);
