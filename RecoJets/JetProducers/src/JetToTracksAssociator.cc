// \class JetToTracksAssociator JetToTracksAssociator.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetToTracksAssociator.cc,v 1.16 2007/07/13 14:24:45 fwyzard Exp $
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetToTracksAssociation.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "JetToTracksAssociator.h"

JetToTracksAssociator::JetToTracksAssociator(const edm::ParameterSet& fConfig)
  : mJets (fConfig.getParameter<edm::InputTag> ("jets")),
    mTracks (fConfig.getParameter<edm::InputTag> ("tracks"))
{
  double dr = fConfig.getParameter<double> ("coneSize");
  mDeltaR2Cut = dr*dr;
  produces<reco::JetToTracksAssociation::Container> ();
}


JetToTracksAssociator::~JetToTracksAssociator() {}

void JetToTracksAssociator::produce(edm::Event& fEvent, const edm::EventSetup& fSetup) {
  edm::Handle <edm::View <reco::Jet> > jets;
  fEvent.getByLabel (mJets, jets);
  edm::Handle <reco::TrackCollection> tracks;
  fEvent.getByLabel (mTracks, tracks);
  
  std::auto_ptr<reco::JetToTracksAssociation::Container> jetTracks (new reco::JetToTracksAssociation::Container);
  // cache tracks kinematics
  std::vector <math::RhoEtaPhiVector> trackP3s;
  trackP3s.reserve (tracks->size());
  for (unsigned i = 0; i < tracks->size(); ++i) {
    const reco::Track* track = &((*(tracks)) [i]);
    trackP3s.push_back (math::RhoEtaPhiVector (track->p(),track->eta(), track->phi())); 
  }
  //loop on jets and associate
  for (unsigned j = 0; j < jets->size(); ++j) {
    reco::TrackRefVector assoTracks;
    double jetEta = (*jets)[j].eta();
    double jetPhi = (*jets)[j].phi();
    for (unsigned t = 0; t < tracks->size(); ++t) {
      double dR2 = deltaR2 (jetEta, jetPhi, trackP3s[t].eta(), trackP3s[t].phi());
      if (dR2 < mDeltaR2Cut)  assoTracks.push_back (reco::TrackRef (tracks, t));
    }
    reco::JetToTracksAssociation::setValue (*jetTracks, jets->refAt(j), assoTracks);
  }
  
  fEvent.put (jetTracks);
}
