#ifndef RecoMuon_MuonIdentification_VersionedMuonSelectors_h
#define RecoMuon_MuonIdentification_VersionedMuonSelectors_h

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"

typedef VersionedSelector<edm::Ptr<reco::Muon> > VersionedRecoMuonSelector;
typedef VersionedSelector<edm::Ptr<pat::Muon > > VersionedPatMuonSelector ;

#endif

