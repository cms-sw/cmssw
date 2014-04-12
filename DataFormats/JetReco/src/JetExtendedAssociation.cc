#include "JetAssociationTemplate.icc"

#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

/// Number of tracks associated in the vertex
int reco::JetExtendedAssociation::tracksAtVertexNumber (const Container& fContainer, const reco::JetBaseRef& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexNumber;
}
int reco::JetExtendedAssociation::tracksAtVertexNumber (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexNumber;
}
/// p4 of tracks associated in the vertex
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtVertexP4 (const Container& fContainer, const reco::JetBaseRef& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexP4;
}
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtVertexP4 (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexP4;
}
/// Number of tracks associated at calo face
int reco::JetExtendedAssociation::tracksAtCaloNumber (const Container& fContainer, const reco::JetBaseRef& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloNumber;
}
int reco::JetExtendedAssociation::tracksAtCaloNumber (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloNumber;
}
/// p4 of tracks associated at calo face
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtCaloP4 (const Container& fContainer, const reco::JetBaseRef& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloP4;
}
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtCaloP4 (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloP4;
}

bool reco::JetExtendedAssociation::setValue (Container* fContainer, 
					    const reco::JetBaseRef& fJet, 
					    const reco::JetExtendedAssociation::JetExtendedData& fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet, fValue);
}

bool reco::JetExtendedAssociation::setValue (Container& fContainer, 
					    const reco::JetBaseRef& fJet, 
					    const reco::JetExtendedAssociation::JetExtendedData& fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet, fValue);
}

const reco::JetExtendedAssociation::JetExtendedData& 
reco::JetExtendedAssociation::getValue (const Container& fContainer, 
					const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::getValue<Container, Value> (fContainer, fJet);
}

const reco::JetExtendedAssociation::JetExtendedData& 
reco::JetExtendedAssociation::getValue (const Container& fContainer, 
					const reco::Jet& fJet) {
  return JetAssociationTemplate::getValue<Container, Value> (fContainer, fJet);
}

std::vector<reco::JetBaseRef > reco::JetExtendedAssociation::allJets (const Container& fContainer) {
  return JetAssociationTemplate::allJets (fContainer);
}
  
bool reco::JetExtendedAssociation::hasJet (const Container& fContainer, 
					  const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}

bool reco::JetExtendedAssociation::hasJet (const Container& fContainer, 
					   const reco::Jet& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}

reco::JetExtendedAssociation::JetExtendedData::JetExtendedData () 
  : mTracksAtVertexNumber (0),
    mTracksAtVertexP4 (0, 0, 0, 0),
    mTracksAtCaloNumber (0),
    mTracksAtCaloP4 (0, 0, 0, 0)
{}
