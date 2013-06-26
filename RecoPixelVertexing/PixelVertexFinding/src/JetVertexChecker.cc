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
// $Id: JetVertexChecker.cc,v 1.2 2013/02/26 21:19:31 chrjones Exp $
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
     edm::InputTag m_associator; 
     edm::InputTag m_primaryVertexProducer;
     edm::InputTag m_beamSpot;
     bool m_doFilter;
     double m_cutMinPt;
     double m_cutMinPtRatio; 
     int32_t m_maxNjets;

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
  m_beamSpot          = iConfig.getParameter<edm::InputTag>("beamSpot");
  m_associator              = iConfig.getParameter<edm::InputTag>("jetTracks");
  m_doFilter                = iConfig.getParameter<bool>("doFilter");
  m_cutMinPt                = iConfig.getParameter<double>("minPt");
  m_cutMinPtRatio           = iConfig.getParameter<double>("minPtRatio");
  m_maxNjets           = iConfig.getParameter<int32_t>("maxNJetsToCheck");
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
//

// ------------ method called on each new Event  ------------
bool
JetVertexChecker::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByLabel(m_associator, jetTracksAssociation);
   std::auto_ptr<std::vector<reco::CaloJet> > pOut(new std::vector<reco::CaloJet> );

   bool result=true;
   int i = 0;
   //limit to first two jets
   for(reco::JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();
       it != jetTracksAssociation->end() && i < m_maxNjets; it++, i++) {
     if(fabs(it->first->eta()) < 2.4)
     {
      reco::TrackRefVector tracks = it->second;
      math::XYZVector jetMomentum = it->first->momentum();
      math::XYZVector trMomentum;
      for(reco::TrackRefVector::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack) 
      {
             trMomentum += (*itTrack)->momentum();
      }
      if(trMomentum.rho()/jetMomentum.rho() < m_cutMinPtRatio || trMomentum.rho() < m_cutMinPt) 
      {
//        std::cout << "bad jet " << it->first->pt() << std::endl;
        pOut->push_back(* dynamic_cast<const reco::CaloJet *>(&(*it->first)));
        result=false;
      }
     }
    } 
  
    iEvent.put(pOut);

   edm::Handle<reco::BeamSpot> beamSpot;
   iEvent.getByLabel(m_beamSpot,beamSpot);
 
   reco::Vertex::Error e;
   e(0, 0) = 0.0015 * 0.0015;
   e(1, 1) = 0.0015 * 0.0015;
   e(2, 2) = 1.5 * 1.5;
   reco::Vertex::Point p(beamSpot->x0(), beamSpot->y0(), beamSpot->z0());
   reco::Vertex thePV(p, e, 0, 0, 0);
   std::auto_ptr<reco::VertexCollection> pOut2(new reco::VertexCollection);
   pOut2->push_back(thePV);
   iEvent.put(pOut2);
//   std::cout << " filter " << result << std::endl;
   if(m_doFilter) return result;
   else 
   return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetVertexChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(JetVertexChecker);
