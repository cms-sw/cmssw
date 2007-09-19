#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

namespace {
  reco::JetTracksAssociation::Container::const_iterator findRef (const reco::JetTracksAssociation::Container& fContainer,
								   const edm::RefToBase<reco::Jet>& fJet) {
    reco::JetTracksAssociation::Container::const_iterator i = fContainer.begin();
    for (; i != fContainer.end(); ++i) {
      if (i->first == fJet) return i;
    }
    return fContainer.end();
  }
  reco::JetTracksAssociation::Container::const_iterator findJet (const reco::JetTracksAssociation::Container& fContainer,
								   const reco::Jet& fJet) {
    reco::JetTracksAssociation::Container::const_iterator i = fContainer.begin();
    for (; i != fContainer.end(); ++i) {
      if (&*(i->first) == &fJet) return i;
    }
    return fContainer.end();
  }
}

/// Get number of tracks associated with jet
int reco::JetTracksAssociation::tracksNumber (const Container& fContainer, const edm::RefToBase<reco::Jet> fJet) {
  return getValue (fContainer, fJet).size();
}
int reco::JetTracksAssociation::tracksNumber (const Container& fContainer, const reco::Jet& fJet) {
  return getValue (fContainer, fJet).size();
}
/// Get LorentzVector as sum of all tracks associated with jet.
reco::JetTracksAssociation::LorentzVector 
reco::JetTracksAssociation::tracksP4 (const Container& fContainer, const edm::RefToBase<reco::Jet> fJet) {
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
					    const edm::RefToBase<reco::Jet>& fJet, 
					    reco::TrackRefVector fValue) {
  if (!fContainer) return false;
  if (findRef (*fContainer, fJet) != fContainer->end ()) return false;
  fContainer->push_back (Container::value_type (fJet, fValue));
  return true;
}

bool reco::JetTracksAssociation::setValue (Container& fContainer, 
					    const edm::RefToBase<reco::Jet>& fJet, 
					    reco::TrackRefVector fValue) {
  return setValue (&fContainer, fJet, fValue);
}

const reco::TrackRefVector& reco::JetTracksAssociation::getValue (const Container& fContainer, 
							    const edm::RefToBase<reco::Jet>& fJet) {
  reco::JetTracksAssociation::Container::const_iterator i = findRef (fContainer, fJet);
  if (i != fContainer.end ()) return i->second;
  throw cms::Exception("No Association") << " in reco::JetTracksAssociation::getValue";
}

const reco::TrackRefVector& reco::JetTracksAssociation::getValue (const Container& fContainer, 
							    const reco::Jet& fJet) {
  reco::JetTracksAssociation::Container::const_iterator i = findJet (fContainer, fJet);
  if (i != fContainer.end ()) return i->second;
  throw cms::Exception("No Association") << " in reco::JetTracksAssociation::getValue";
}

std::vector<edm::RefToBase<reco::Jet> > reco::JetTracksAssociation::allJets (const Container& fContainer) {
  std::vector<edm::RefToBase<reco::Jet> > result;
  reco::JetTracksAssociation::Container::const_iterator i = fContainer.begin();
  for (; i != fContainer.end(); ++i) {
    result.push_back (i->first);
  }
  return result;
}
  
bool reco::JetTracksAssociation::hasJet (const Container& fContainer, 
					  const edm::RefToBase<reco::Jet>& fJet) {
  return findRef (fContainer, fJet) != fContainer.end();
}
