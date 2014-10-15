// -*- C++ -*-
//
// Package:    SwitchTrackCollection
// Class:      SwitchTrackCollection
// 
/**\class SwitchTrackCollection SwitchTrackCollection.cc RecoBTag/SwitchTrackCollection/src/SwitchTrackCollection.cc

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
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//

class SwitchTrackCollection : public edm::EDProducer {
   public:
      explicit SwitchTrackCollection(const edm::ParameterSet&);
      ~SwitchTrackCollection();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
     edm::EDGetTokenT<std::vector<reco::Track> > m_track1; 
     edm::EDGetTokenT<std::vector<reco::Track> > m_track2; 

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
SwitchTrackCollection::SwitchTrackCollection(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  m_track1              = consumes<std::vector<reco::Track> >(iConfig.getParameter<edm::InputTag>("TrackCollection1"));
  m_track2              = consumes<std::vector<reco::Track> >(iConfig.getParameter<edm::InputTag>("TrackCollection2"));

  produces<std::vector<reco::Track> >(); 

//  m_beamSpot          = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"));
//  m_associator              = consumes<reco::JetTracksAssociationCollection>(iConfig.getParameter<edm::InputTag>("jetTracks"));
//  m_doFilter                = iConfig.getParameter<bool>("doFilter");
//  m_cutMinPt                = iConfig.getParameter<double>("minPt");
//  m_cutMinPtRatio           = iConfig.getParameter<double>("minPtRatio");
//  m_maxNjets           = iConfig.getParameter<int32_t>("maxNJetsToCheck");
//  m_newMethod                = iConfig.getParameter<bool>("newMethod");
//  produces<std::vector<reco::CaloJet> >(); 
//  produces<reco::VertexCollection >(); 
//  produces<float >(); 
//  produces<float >("ntrks"); 
//  produces<int >("trkPt"); 
}


SwitchTrackCollection::~SwitchTrackCollection()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
void SwitchTrackCollection::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   
   Handle<std::vector<reco::Track> > tracks1;
   iEvent.getByToken(m_track1, tracks1);
   
   Handle<std::vector<reco::Track> > tracks2;
   iEvent.getByToken(m_track2, tracks2);

   std::auto_ptr<std::vector<reco::Track> > pOut(new std::vector<reco::Track> );

   if(tracks2->size()==0)	*pOut = *tracks1;
   else	*pOut = *tracks2;
   
   iEvent.put(pOut);
   
//   return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SwitchTrackCollection::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(SwitchTrackCollection);
