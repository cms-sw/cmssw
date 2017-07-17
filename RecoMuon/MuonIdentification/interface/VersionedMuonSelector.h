#ifndef __RecoMuon_MuonIdentification_VersionedMuonSelector__
#define __RecoMuon_MuonIdentification_VersionedMuonSelector__

// user include files
#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

typedef VersionedSelector<edm::Ptr<reco::Muon> > VersionedMuonSelector;

#endif
