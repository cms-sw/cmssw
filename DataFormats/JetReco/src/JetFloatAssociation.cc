#include "JetAssociationTemplate.icc"

#include "DataFormats/JetReco/interface/JetFloatAssociation.h"


bool reco::JetFloatAssociation::setValue (Container* fContainer, 
					    const reco::JetBaseRef& fJet, 
					    float fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet, fValue);
}

bool reco::JetFloatAssociation::setValue (Container& fContainer, 
					    const reco::JetBaseRef& fJet, 
					    float fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet, fValue);
}

float reco::JetFloatAssociation::getValue (const Container& fContainer, 
					     const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::getValue<Container, Value> (fContainer, fJet);
}

float reco::JetFloatAssociation::getValue (const Container& fContainer, 
					     const reco::Jet& fJet) {
  return JetAssociationTemplate::getValue<Container, Value> (fContainer, fJet);
}

std::vector<reco::JetBaseRef > reco::JetFloatAssociation::allJets (const Container& fContainer) {
  return JetAssociationTemplate::allJets (fContainer);
}
  
bool reco::JetFloatAssociation::hasJet (const Container& fContainer, 
					const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}

bool reco::JetFloatAssociation::hasJet (const Container& fContainer, 
					const reco::Jet& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}
