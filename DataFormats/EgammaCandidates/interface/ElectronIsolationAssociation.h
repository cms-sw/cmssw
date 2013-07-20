#ifndef EgammaCandidates_ElectronIsolationAssociation_h
#define EgammaCandidates_ElectronIsolationAssociation_h
// \class ElectronIsolationAssociation
// 
// \short association of Isolation to an Electron
// $Id: ElectronIsolationAssociation.h,v 1.2 2007/07/31 15:20:03 ratnik Exp $
//

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h" 
#include <vector>

namespace reco {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::Electron>, float > > ElectronIsolationMap;
}
#endif
