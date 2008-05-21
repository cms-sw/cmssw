#include "JetAssociationTemplate.icc"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

/// Get number of tracks associated with jet
int reco::JetTracksAssociation::tracksNumber (const Container& fContainer, const reco::JetBaseRef fJet) {
  return getValue (fContainer, fJet).size();
}
int reco::JetTracksAssociation::tracksNumber (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).size();
}
/// Get LorentzVector as sum of all tracks associated with jet.
reco::JetTracksAssociation::LorentzVector 
reco::JetTracksAssociation::tracksP4 (const Container& fContainer, const reco::JetBaseRef fJet) {
  const reco::TrackRefVector* tracks = &getValue (fContainer, fJet);
  math::XYZTLorentzVector result (0,0,0,0);
  for (unsigned t = 0; t < tracks->size(); ++t) {
    const reco::Track& track = *((*tracks)[t]);
    result += math::XYZTLorentzVector (track.px(), track.py(), track.pz(), track.p()); // massless hypothesis 
  }
  return reco::JetTracksAssociation::LorentzVector (result);
}
reco::JetTracksAssociation::LorentzVector 
reco::JetTracksAssociation::tracksP4 (const Container& fContainer, const reco::Jet& fJet) {
  const reco::TrackRefVector* tracks = &getValue (fContainer, fJet);
  math::XYZTLorentzVector result (0,0,0,0);
  for (unsigned t = 0; t < tracks->size(); ++t) {
    const reco::Track& track = *((*tracks)[t]);
    result += math::XYZTLorentzVector (track.px(), track.py(), track.pz(), track.p()); // massless hypothesis 
  }
  return reco::JetTracksAssociation::LorentzVector (result);
}


bool reco::JetTracksAssociation::setValue (Container* fContainer, 
					   const reco::JetBaseRef& fJet, 
					   reco::TrackRefVector fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet,fValue);
}

bool reco::JetTracksAssociation::setValue (Container& fContainer, 
					    const reco::JetBaseRef& fJet, 
					    reco::TrackRefVector fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet,fValue);
}

const reco::TrackRefVector& reco::JetTracksAssociation::getValue (const Container& fContainer, 
							    const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::getValue<Container, reco::TrackRefVector> (fContainer, fJet);
}

const reco::TrackRefVector& reco::JetTracksAssociation::getValue (const Container& fContainer, 
							    const reco::Jet& fJet) {
  return JetAssociationTemplate::getValue<Container, reco::TrackRefVector> (fContainer, fJet);
}

std::vector<reco::JetBaseRef > reco::JetTracksAssociation::allJets (const Container& fContainer) {
  return JetAssociationTemplate::allJets (fContainer);
}
  
bool reco::JetTracksAssociation::hasJet (const Container& fContainer, 
					  const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}

bool reco::JetTracksAssociation::hasJet (const Container& fContainer, 
					 const reco::Jet& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}
