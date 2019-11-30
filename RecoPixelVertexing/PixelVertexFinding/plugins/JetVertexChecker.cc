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
#include "FWCore/Framework/interface/stream/EDFilter.h"

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

class JetVertexChecker : public edm::stream::EDFilter<> {
public:
  explicit JetVertexChecker(const edm::ParameterSet&);
  ~JetVertexChecker() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<reco::JetTracksAssociationCollection> m_associator;
  const edm::EDGetTokenT<reco::BeamSpot> m_beamSpot;
  const bool m_doFilter;
  const double m_cutMinPt;
  const double m_cutMinPtRatio;
  const double m_maxTrackPt;
  const double m_maxChi2;
  const int32_t m_maxNjets;
  const int32_t m_maxNjetsOutput;

  const bool m_newMethod;

  const double m_maxETA;
  const double m_pvErr_x;
  const double m_pvErr_y;
  const double m_pvErr_z;
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
    : m_associator(consumes<reco::JetTracksAssociationCollection>(iConfig.getParameter<edm::InputTag>("jetTracks"))),
      m_beamSpot(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      m_doFilter(iConfig.getParameter<bool>("doFilter")),
      m_cutMinPt(iConfig.getParameter<double>("minPt")),
      m_cutMinPtRatio(iConfig.getParameter<double>("minPtRatio")),
      m_maxTrackPt(iConfig.getParameter<double>("maxTrackPt")),
      m_maxChi2(iConfig.getParameter<double>("maxChi2")),
      m_maxNjets(iConfig.getParameter<int32_t>("maxNJetsToCheck")),
      m_maxNjetsOutput(iConfig.getParameter<int32_t>("maxNjetsOutput")),
      m_newMethod(iConfig.getParameter<bool>("newMethod")),
      m_maxETA(iConfig.getParameter<double>("maxETA")),
      m_pvErr_x(iConfig.getParameter<double>("pvErr_x")),
      m_pvErr_y(iConfig.getParameter<double>("pvErr_y")),
      m_pvErr_z(iConfig.getParameter<double>("pvErr_z")) {
  //now do what ever initialization is needed
  produces<std::vector<reco::CaloJet>>();
  produces<reco::VertexCollection>();
}

JetVertexChecker::~JetVertexChecker() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//	m_maxChi2 m_maxTrackPt

// ------------ method called on each new Event  ------------
bool JetVertexChecker::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
  iEvent.getByToken(m_associator, jetTracksAssociation);
  auto pOut = std::make_unique<std::vector<reco::CaloJet>>();

  bool result = true;
  int i = 0;
  float calopt = 0;
  float trkpt = 0;
  //limit to first two jets
  for (reco::JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin(),
                                                            et = jetTracksAssociation->end();
       it != et && i < m_maxNjets;
       it++, i++) {
    if (std::abs(it->first->eta()) < m_maxETA) {
      reco::TrackRefVector tracks = it->second;
      math::XYZVector jetMomentum = it->first->momentum();
      math::XYZVector trMomentum;
      for (reco::TrackRefVector::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack) {
        const reco::Track& iTrack = **itTrack;
        if (m_newMethod && iTrack.chi2() > m_maxChi2)
          continue;
        trMomentum += iTrack.momentum();
        if (m_newMethod)
          trkpt += std::min(m_maxTrackPt, (iTrack.pt()));
        else
          trkpt += iTrack.pt();
      }
      calopt += jetMomentum.rho();
      if (trMomentum.rho() / jetMomentum.rho() < m_cutMinPtRatio || trMomentum.rho() < m_cutMinPt) {
        pOut->push_back(*dynamic_cast<const reco::CaloJet*>(&(*it->first)));
      }
    }
  }
  iEvent.put(std::move(pOut));

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(m_beamSpot, beamSpot);

  reco::Vertex::Error e;
  e(0, 0) = m_pvErr_x * m_pvErr_x;
  e(1, 1) = m_pvErr_y * m_pvErr_y;
  e(2, 2) = m_pvErr_z * m_pvErr_z;
  reco::Vertex::Point p(beamSpot->x0(), beamSpot->y0(), beamSpot->z0());
  reco::Vertex thePV(p, e, 0, 0, 0);
  auto pOut2 = std::make_unique<reco::VertexCollection>();
  pOut2->push_back(thePV);
  iEvent.put(std::move(pOut2));

  if (m_doFilter)
    return result;
  else
    return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void JetVertexChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("jetTracks", edm::InputTag("hltFastPVJetTracksAssociator"));
  desc.add<double>("minPtRatio", 0.1);
  desc.add<double>("minPt", 0.0);
  desc.add<bool>("doFilter", false);
  desc.add<int>("maxNJetsToCheck", 2);
  desc.add<int>("maxNjetsOutput", 2);
  desc.add<double>("maxChi2", 20.0);
  desc.add<double>("maxTrackPt", 20.0);
  desc.add<bool>("newMethod", false);  // <---- newMethod
  desc.add<double>("maxETA", 2.4);
  desc.add<double>("pvErr_x", 0.0015);
  desc.add<double>("pvErr_y", 0.0015);
  desc.add<double>("pvErr_z", 1.5);
  descriptions.add("jetVertexChecker", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(JetVertexChecker);
