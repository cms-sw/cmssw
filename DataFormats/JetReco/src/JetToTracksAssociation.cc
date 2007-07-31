#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/JetReco/interface/JetToTracksAssociation.h"

namespace {
  reco::JetToTracksAssociation::Container::const_iterator findRef (const reco::JetToTracksAssociation::Container& fContainer,
								  const edm::RefToBase<reco::Jet>& fJet) {
    reco::JetToTracksAssociation::Container::const_iterator i = fContainer.begin();
    for (; i != fContainer.end(); ++i) {
      if (i->first == fJet) return i;
    }
    return fContainer.end();
  }
}


bool reco::JetToTracksAssociation::setValue (Container* fContainer, 
					    const edm::RefToBase<reco::Jet>& fJet, 
					    reco::TrackRefVector fValue) {
  if (!fContainer) return false;
  if (findRef (*fContainer, fJet) != fContainer->end ()) return false;
  fContainer->push_back (Container::value_type (fJet, fValue));
  return true;
}

bool reco::JetToTracksAssociation::setValue (Container& fContainer, 
					    const edm::RefToBase<reco::Jet>& fJet, 
					    reco::TrackRefVector fValue) {
  return setValue (&fContainer, fJet, fValue);
}

reco::TrackRefVector reco::JetToTracksAssociation::getValue (const Container& fContainer, 
							    const edm::RefToBase<reco::Jet>& fJet) {
  reco::JetToTracksAssociation::Container::const_iterator i = findRef (fContainer, fJet);
  if (i != fContainer.end ()) return i->second;
  throw cms::Exception("No Association") << " in reco::JetToTracksAssociation::getValue";
}

std::vector<edm::RefToBase<reco::Jet> > reco::JetToTracksAssociation::allJets (const Container& fContainer) {
  std::vector<edm::RefToBase<reco::Jet> > result;
  reco::JetToTracksAssociation::Container::const_iterator i = fContainer.begin();
  for (; i != fContainer.end(); ++i) {
    result.push_back (i->first);
  }
  return result;
}
  
bool reco::JetToTracksAssociation::hasJet (const Container& fContainer, 
					  const edm::RefToBase<reco::Jet>& fJet) {
  return findRef (fContainer, fJet) != fContainer.end();
}
