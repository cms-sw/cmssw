// \class JetTracksAssociatorAtVertex JetTracksAssociatorAtVertex.cc 
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
// $Id: JetTracksAssociatorAtVertex.cc,v 1.2 2007/10/05 23:23:11 fedor Exp $
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "JetTracksAssociatorAtVertex.h"

JetTracksAssociatorAtVertex::JetTracksAssociatorAtVertex(const edm::ParameterSet& fConfig)
  : mJets (fConfig.getParameter<edm::InputTag> ("jets")),
    mTracks (fConfig.getParameter<edm::InputTag> ("tracks")),
    mAssociator (fConfig.getParameter<double> ("coneSize"))
{
  reco::TrackBase::TrackQuality trackQuality = 
    reco::TrackBase::qualityByName (fConfig.getParameter<std::string> ("trackQuality"));
  if (trackQuality == reco::TrackBase::undefQuality) { // we have a problem
    edm::LogError("JetTracksAssociatorAtVertex") << "Unknown trackQuality value '" 
						 << fConfig.getParameter<std::string> ("trackQuality")
						 << "'. See possible values in 'reco::TrackBase::qualityByName'";
  }
  mTrackQuality = int (trackQuality);
  produces<reco::JetTracksAssociation::Container> ();
}
  produces<reco::JetTracksAssociation::Container> ();
}

JetTracksAssociatorAtVertex::~JetTracksAssociatorAtVertex() {}

void JetTracksAssociatorAtVertex::produce(edm::Event& fEvent, const edm::EventSetup& fSetup) {
  edm::Handle <edm::View <reco::Jet> > jets_h;
  fEvent.getByLabel (mJets, jets_h);
  edm::Handle <reco::TrackCollection> tracks_h;
  fEvent.getByLabel (mTracks, tracks_h);
  
  std::auto_ptr<reco::JetTracksAssociation::Container> jetTracks (new reco::JetTracksAssociation::Container (reco::JetRefBaseProd(jets_h)));

  // format inputs
  std::vector <edm::RefToBase<reco::Jet> > allJets;
  allJets.reserve (jets_h->size());
  for (unsigned i = 0; i < jets_h->size(); ++i) allJets.push_back (jets_h->refAt(i));
  std::vector <reco::TrackRef> allTracks;
  allTracks.reserve (tracks_h->size());
  reco::TrackBase::TrackQuality trackQuality = reco::TrackBase::TrackQuality (mTrackQuality); // convert back
  // run algo
  for (unsigned i = 0; i < tracks_h->size(); ++i) {
    if ((*tracks_h)[i].quality (trackQuality)) allTracks.push_back (reco::TrackRef (tracks_h, i));
  }
  mAssociator.produce (&*jetTracks, allJets, allTracks);
  // store output
  fEvent.put (jetTracks);
}
