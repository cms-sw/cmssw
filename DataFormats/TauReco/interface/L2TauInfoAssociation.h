/*
Association Map Definition (Jets - L2 Info)
for L2 Tau Trigger

Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/

#include "DataFormats/TauReco/interface/L2TauIsolationInfo.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include <vector>

namespace reco {
typedef edm::AssociationMap< edm::OneToValue< reco::CaloJetCollection, reco::L2TauIsolationInfo >  > L2TauInfoAssociation; 
}
