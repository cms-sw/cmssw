#ifndef EgammaCandidates_ElectronIsolationAssociation_h
#define EgammaCandidates_ElectronIsolationAssociation_h
// \class ElectronIsolationAssociation
// 
// \short association of Isolation to an Electron
// $Id: ElectronIsolationAssociation.h,v 1.1 2006/10/20 12:59:03 rahatlou Exp $
//

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h" 
#include <vector>

namespace reco {
  typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::Electron>, float > > ElectronIsolationMap;
}
#endif
