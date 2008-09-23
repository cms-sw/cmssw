#ifndef DataFormats_TauReco_PFTauDecayModeAssociation_h
#define DataFormats_TauReco_PFTauDecayModeAssociation_h

#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/Common/interface/Association.h"

namespace reco {
   typedef edm::Association<reco::PFTauDecayModeCollection> PFTauDecayModeMatchMap;
}

#endif
