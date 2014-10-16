// -*- C++ -*-
//
// Package:    JetVertexChecker
// Class:      JetVertexChecker
// 
/**\class JetVertexChecker JetVertexChecker.cc RecoBTag/JetVertexChecker/src/JetVertexChecker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrea RIZZI
//         Created:  Mon Jan 16 11:19:48 CET 2012
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

//
// class declaration
//

class JetVertexChecker : public edm::EDFilter {
   public:
      explicit JetVertexChecker(const edm::ParameterSet&);
      ~JetVertexChecker();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
     edm::EDGetTokenT<reco::JetTracksAssociationCollection> m_associator; 
     edm::EDGetTokenT<reco::BeamSpot> m_beamSpot;
     bool m_doFilter;
     double m_cutMinPt;
     double m_cutMinPtRatio; 
     double m_maxTrackPt;
     double m_maxChi2;
     int32_t m_maxNjets;
     int32_t m_maxNjetsOutput;
     

     bool m_newMethod;

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
JetVertexChecker::JetVertexChecker(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  m_beamSpot          	    = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"));
  m_associator              = consumes<reco::JetTracksAssociationCollection>(iConfig.getParameter<edm::InputTag>("jetTracks"));
  m_doFilter                = iConfig.getParameter<bool>("doFilter");
  m_cutMinPt                = iConfig.getParameter<double>("minPt");
  m_cutMinPtRatio           = iConfig.getParameter<double>("minPtRatio");
  m_maxNjets         	    = iConfig.getParameter<int32_t>("maxNJetsToCheck");
  m_maxNjetsOutput          = iConfig.getParameter<int32_t>("maxNjetsOutput");
  m_newMethod                = iConfig.getParameter<bool>("newMethod");
  m_maxTrackPt           = iConfig.getParameter<double>("maxTrackPt");
  m_maxChi2           = iConfig.getParameter<double>("maxChi2");
  produces<std::vector<reco::CaloJet> >(); 
  produces<reco::VertexCollection >(); 
}


JetVertexChecker::~JetVertexChecker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//	m_maxChi2 m_maxTrackPt

// ------------ method called on each new Event  ------------
bool
JetVertexChecker::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{  
   using namespace edm;
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByToken(m_associator, jetTracksAssociation);
   std::auto_ptr<std::vector<reco::CaloJet> > pOut(new std::vector<reco::CaloJet> );

   bool result=true;
   int i = 0;
   float calopt=0;
   float trkpt=0;
   //limit to first two jets
   for(reco::JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();
       it != jetTracksAssociation->end() && i < m_maxNjets; it++, i++) {
     if(std::abs(it->first->eta()) < 2.4)
     {
      reco::TrackRefVector tracks = it->second;
      math::XYZVector jetMomentum = it->first->momentum();
      math::XYZVector trMomentum;
      for(reco::TrackRefVector::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack) 
      {
         const reco::Track& iTrack = **itTrack;
	     if(m_newMethod && iTrack.chi2()>m_maxChi2) continue;
             trMomentum += iTrack.momentum();
	     if(m_newMethod) trkpt += std::min(m_maxTrackPt,( iTrack.pt()));
	     else trkpt += iTrack.pt();
      }
      calopt += jetMomentum.rho();
      if(trMomentum.rho()/jetMomentum.rho() < m_cutMinPtRatio || trMomentum.rho() < m_cutMinPt) 
      {
        pOut->push_back(* dynamic_cast<const reco::CaloJet *>(&(*it->first)));
      }
     }
    }
   iEvent.put(pOut);

   edm::Handle<reco::BeamSpot> beamSpot;
   iEvent.getByToken(m_beamSpot,beamSpot);
 
   reco::Vertex::Error e;
   e(0, 0) = 0.0015 * 0.0015;
   e(1, 1) = 0.0015 * 0.0015;
   e(2, 2) = 1.5 * 1.5;
   reco::Vertex::Point p(beamSpot->x0(), beamSpot->y0(), beamSpot->z0());
   reco::Vertex thePV(p, e, 0, 0, 0);
   std::auto_ptr<reco::VertexCollection> pOut2(new reco::VertexCollection);
   pOut2->push_back(thePV);
   iEvent.put(pOut2);

   if(m_doFilter) return result;
   else 
   return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetVertexChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   edm::ParameterSetDescription desc;
   desc.add<edm::InputTag> ("beamSpot",edm::InputTag("hltOnlineBeamSpot"));
   desc.add<edm::InputTag> ("jetTracks",edm::InputTag("hltFastPVJetTracksAssociator"));
   desc.add<double> ("minPtRatio",0.1);
   desc.add<double> ("minPt",0.0);
   desc.add<bool> ("doFilter",false);
   desc.add<int> ("maxNJetsToCheck",2);
   desc.add<int> ("maxNjetsOutput",2);
   desc.add<double> ("maxChi2",20.0);
   desc.add<double> ("maxTrackPt",20.0);
   desc.add<bool> ("newMethod",false);		// <---- newMethod 
   descriptions.add("jetVertexChecker",desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(JetVertexChecker);
