#include "JetAssociationTemplate.icc"

#include "DataFormats/JetReco/interface/JetToFloatAssociation.h"


bool reco::JetToFloatAssociation::setValue (Container* fContainer, 
					    const reco::JetBaseRef& fJet, 
					    float fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet, fValue);
}

bool reco::JetToFloatAssociation::setValue (Container& fContainer, 
					    const reco::JetBaseRef& fJet, 
					    float fValue) {
  return JetAssociationTemplate::setValue (fContainer, fJet, fValue);
}

float reco::JetToFloatAssociation::getValue (const Container& fContainer, 
					     const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::getValue<Container, Value> (fContainer, fJet);
}

float reco::JetToFloatAssociation::getValue (const Container& fContainer, 
					     const reco::Jet& fJet) {
  return JetAssociationTemplate::getValue<Container, Value> (fContainer, fJet);
}

std::vector<reco::JetBaseRef > reco::JetToFloatAssociation::allJets (const Container& fContainer) {
  return JetAssociationTemplate::allJets (fContainer);
}
  
bool reco::JetToFloatAssociation::hasJet (const Container& fContainer, 
					const reco::JetBaseRef& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}

bool reco::JetToFloatAssociation::hasJet (const Container& fContainer, 
					const reco::Jet& fJet) {
  return JetAssociationTemplate::hasJet (fContainer, fJet);
}

const reco::JetToFloatAssociation::Container* reco::JetToFloatAssociation::getByLabel (const fwlite::Event& fEvent, 
										   const char* fModuleLabel,
										   const char* fProductInstanceLabel,
										   const char* fProcessLabel) {
  return JetAssociationTemplate::getByLabel<Container> (fEvent, fModuleLabel, fProductInstanceLabel, fProcessLabel);
}
