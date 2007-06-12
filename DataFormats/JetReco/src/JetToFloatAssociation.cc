#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/JetReco/interface/JetToFloatAssociation.h"

namespace {
  reco::JetToFloatAssociation::Container::const_iterator findRef (const reco::JetToFloatAssociation::Container& fContainer,
								  const edm::RefToBase<reco::Jet>& fJet) {
    reco::JetToFloatAssociation::Container::const_iterator i = fContainer.begin();
    for (; i != fContainer.end(); ++i) {
      if (i->first == fJet) return i;
    }
    return fContainer.end();
  }
}


bool reco::JetToFloatAssociation::setValue (Container& fContainer, 
					    const edm::RefToBase<reco::Jet>& fJet, 
					    float fValue) {
  if (findRef (fContainer, fJet) == fContainer.end ()) return false;
  fContainer.push_back (Container::value_type (fJet, fValue));
  return true;
}

float reco::JetToFloatAssociation::getValue (const Container& fContainer, 
					     const edm::RefToBase<reco::Jet>& fJet) {
  reco::JetToFloatAssociation::Container::const_iterator i = findRef (fContainer, fJet);
  if (i != fContainer.end ()) return i->second;
  throw cms::Exception("No Association") << " in reco::JetToFloatAssociation::getValue";
}

std::vector<edm::RefToBase<reco::Jet> > reco::JetToFloatAssociation::allJets (const Container& fContainer) {
  std::vector<edm::RefToBase<reco::Jet> > result;
  reco::JetToFloatAssociation::Container::const_iterator i = fContainer.begin();
  for (; i != fContainer.end(); ++i) {
    result.push_back (i->first);
  }
  return result;
}

bool reco::JetToFloatAssociation::hasJet (const Container& fContainer, 
					  const edm::RefToBase<reco::Jet>& fJet) {
  return findRef (fContainer, fJet) != fContainer.end();
}
