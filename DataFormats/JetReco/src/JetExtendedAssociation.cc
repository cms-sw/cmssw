#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/FWLite/interface/Handle.h"

#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

namespace {
  inline unsigned findJet (const reco::JetExtendedAssociation::Container& fContainer,
			   const reco::Jet& fJet) {
    std::cout << "findJet-> " << std::endl;
    std::cout << "findJet-> size: " <<  fContainer.size () << '/' << fContainer.keyProduct().id() << std::endl;
    for (unsigned i = 0; i < fContainer.size (); ++i) {
      std::cout << "findJet-> " << i << '/' << &fJet << '/' << fContainer.key (i).id() << '/' <<  fContainer.key (i).key() << std::endl;
      if (&*(fContainer.key (i)) == &fJet) return i;
    }
    throw cms::Exception("Invalid jet") 
         << "JetExtendedAssociation-> inquire association with jet which is not available in the original jet collection";
  }
}

/// Number of tracks associated in the vertex
int reco::JetExtendedAssociation::tracksAtVertexNumber (const Container& fContainer, const edm::RefToBase<reco::Jet>& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexNumber;
}
int reco::JetExtendedAssociation::tracksAtVertexNumber (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexNumber;
}
/// p4 of tracks associated in the vertex
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtVertexP4 (const Container& fContainer, const edm::RefToBase<reco::Jet>& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexP4;
}
const reco::JetExtendedAssociation::LorentzVector& 
reco::JetExtendedAssociation::tracksAtVertexP4 (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).mTracksAtVertexP4;
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
  fContainer->setValue (fJet.key(), fValue);
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
  return fContainer[fJet];
}

const reco::JetExtendedAssociation::JetExtendedData& 
reco::JetExtendedAssociation::getValue (const Container& fContainer, 
					const reco::Jet& fJet) {
  unsigned i = findJet (fContainer, fJet);
  return fContainer[i].second;
}

std::vector<edm::RefToBase<reco::Jet> > reco::JetExtendedAssociation::allJets (const Container& fContainer) {
  std::vector<edm::RefToBase<reco::Jet> > result;
  for (unsigned i = 0; i < fContainer.size(); ++i) {
    result.push_back (fContainer.key (i));
  }
  return result;
}
  
bool reco::JetExtendedAssociation::hasJet (const Container& fContainer, 
					  const edm::RefToBase<reco::Jet>& fJet) {
  try {
    fContainer [fJet];
    return true;
  }
  catch (cms::Exception e) { // jet not found
    return false;
  }
}

const reco::JetExtendedAssociation::Container* reco::JetExtendedAssociation::getByLabel (const fwlite::Event& fEvent, 
											 const char* fModuleLabel,
											 const char* fProductInstanceLabel,
											 const char* fProcessLabel) {
  fwlite::Handle<reco::JetExtendedAssociation::Container> handle;
  handle.getByLabel (fEvent, fModuleLabel, fProductInstanceLabel, fProcessLabel);
  std::cout << "reco::JetExtendedAssociation::getByLabel->" 
	    << " size: " << handle->size()
	    << ", product ID: " << handle->keyProduct().id()
	    << std::endl;
  return &*handle;
}


reco::JetExtendedAssociation::JetExtendedData::JetExtendedData () 
  : mTracksAtVertexNumber (0),
    mTracksAtVertexP4 (0, 0, 0, 0),
    mTracksAtCaloNumber (0),
    mTracksAtCaloP4 (0, 0, 0, 0)
{}
