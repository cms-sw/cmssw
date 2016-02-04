#ifndef ElectronIDAssociation_h
#define ElectronIDAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {

  // association map
  typedef edm::AssociationMap<edm::OneToOne<GsfElectronCollection, ElectronIDCollection> > ElectronIDAssociationCollection;
 
  typedef ElectronIDAssociationCollection::value_type ElectronIDAssociation;

  /// reference to an object in a collection of SeedMap objects
  typedef edm::Ref<ElectronIDAssociationCollection> ElectronIDAssociationRef;

  /// reference to a collection of SeedMap objects
  typedef edm::RefProd<ElectronIDAssociationCollection> ElectronIDAssociationRefProd;

  /// vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<ElectronIDAssociationCollection> ElectronIDAssociationRefVector;

}

#endif
