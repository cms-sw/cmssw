#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

namespace {
  reco::JetExtendedAssociation::Container::const_iterator findRef (const reco::JetExtendedAssociation::Container& fContainer,
								  const edm::RefToBase<reco::Jet>& fJet) {
    reco::JetExtendedAssociation::Container::const_iterator i = fContainer.begin();
    for (; i != fContainer.end(); ++i) {
      if (i->first == fJet) return i;
    }
    return fContainer.end();
  }
  reco::JetExtendedAssociation::Container::iterator findRef (reco::JetExtendedAssociation::Container* fContainer,
								  const edm::RefToBase<reco::Jet>& fJet) {
    reco::JetExtendedAssociation::Container::iterator i = fContainer->begin();
    for (; i != fContainer->end(); ++i) {
      if (i->first == fJet) return i;
    }
    return fContainer->end();
  }
  reco::JetExtendedAssociation::Container::const_iterator findJet (const reco::JetExtendedAssociation::Container& fContainer,
								  const reco::Jet& fJet) {
    reco::JetExtendedAssociation::Container::const_iterator i = fContainer.begin();
    for (; i != fContainer.end(); ++i) {
      if (&*(i->first) == &fJet) return i;
    }
    return fContainer.end();
  }
}

/// Number of tracks associated in the vertex
int reco::JetExtendedAssociation::tracksInVertexNumber (const Container& fContainer, const edm::RefToBase<reco::Jet>& fJet) {
  return getValue (fContainer, fJet).mTracksInVertexNumber;
}
int reco::JetExtendedAssociation::tracksInVertexNumber (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksInVertexNumber;
}
/// p4 of tracks associated in the vertex
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksInVertexP4 (const Container& fContainer, const edm::RefToBase<reco::Jet>& fJet) {
  return getValue (fContainer, fJet).mTracksInVertexP4;
}
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksInVertexP4 (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksInVertexP4;
}
/// Number of tracks associated at calo face
int reco::JetExtendedAssociation::tracksAtCaloNumber (const Container& fContainer, const edm::RefToBase<reco::Jet>& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloNumber;
}
int reco::JetExtendedAssociation::tracksAtCaloNumber (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloNumber;
}
/// p4 of tracks associated at calo face
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtCaloP4 (const Container& fContainer, const edm::RefToBase<reco::Jet>& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloP4;
}
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtCaloP4 (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtCaloP4;
}

bool reco::JetExtendedAssociation::setValue (Container* fContainer, 
					    const edm::RefToBase<reco::Jet>& fJet, 
					    const reco::JetExtendedAssociation::JetExtendedData& fValue) {
  if (!fContainer) return false;
  if (findRef (*fContainer, fJet) != fContainer->end ()) return false;
  fContainer->push_back (Container::value_type (fJet, fValue));
  return true;
}

bool reco::JetExtendedAssociation::setValue (Container& fContainer, 
					    const edm::RefToBase<reco::Jet>& fJet, 
					    const reco::JetExtendedAssociation::JetExtendedData& fValue) {
  return setValue (&fContainer, fJet, fValue);
}

const reco::JetExtendedAssociation::JetExtendedData& 
reco::JetExtendedAssociation::getValue (const Container& fContainer, 
					const edm::RefToBase<reco::Jet>& fJet) {
  reco::JetExtendedAssociation::Container::const_iterator i = findRef (fContainer, fJet);
  if (i != fContainer.end ()) return i->second;
  throw cms::Exception("No Association") << " in reco::JetExtendedAssociation::getValue";
}

const reco::JetExtendedAssociation::JetExtendedData& 
reco::JetExtendedAssociation::getValue (const Container& fContainer, 
					const reco::Jet& fJet) {
  reco::JetExtendedAssociation::Container::const_iterator i = findJet (fContainer, fJet);
  if (i != fContainer.end ()) return i->second;
  throw cms::Exception("No Association") << " in reco::JetExtendedAssociation::getValue";
}

reco::JetExtendedAssociation::JetExtendedData*
reco::JetExtendedAssociation::getValue (Container* fContainer, 
					const edm::RefToBase<reco::Jet>& fJet) {
  reco::JetExtendedAssociation::Container::iterator i = findRef (fContainer, fJet);
  if (i != fContainer->end ()) return &(i->second);
  return 0;
}

std::vector<edm::RefToBase<reco::Jet> > reco::JetExtendedAssociation::allJets (const Container& fContainer) {
  std::vector<edm::RefToBase<reco::Jet> > result;
  reco::JetExtendedAssociation::Container::const_iterator i = fContainer.begin();
  for (; i != fContainer.end(); ++i) {
    result.push_back (i->first);
  }
  return result;
}
  
bool reco::JetExtendedAssociation::hasJet (const Container& fContainer, 
					  const edm::RefToBase<reco::Jet>& fJet) {
  return findRef (fContainer, fJet) != fContainer.end();
}

reco::JetExtendedAssociation::JetExtendedData::JetExtendedData () 
  : mTracksInVertexNumber (0),
    mTracksInVertexP4 (0, 0, 0, 0),
    mTracksAtCaloNumber (0),
    mTracksAtCaloP4 (0, 0, 0, 0)
{}
