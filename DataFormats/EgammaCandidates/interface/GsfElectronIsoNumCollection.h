#ifndef EgammaCandidates_GsfElectronIsoNumCollection_h
#define EgammaCandidates_GsfElectronIsoNumCollection_h

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include <vector>

namespace reco {

  typedef edm::AssociationVector<reco::GsfElectronRefProd, std::vector<int> > GsfElectronIsoNumCollection;

  typedef GsfElectronIsoNumCollection::value_type GsfElectronIsoNum;
  typedef edm::Ref<GsfElectronIsoNumCollection> GsfElectronIsoNumCollectionRef;
  typedef edm::RefProd<GsfElectronIsoNumCollection> GsfElectronIsoNumCollectionRefProd;
  typedef edm::RefVector<GsfElectronIsoNumCollection> GsfElectronIsoNumCollectionRefVector;

}  // namespace reco

#endif
