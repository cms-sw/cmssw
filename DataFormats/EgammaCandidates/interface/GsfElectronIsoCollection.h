#ifndef EgammaCandidates_GsfElectronIsoCollection_h
#define EgammaCandidates_GsfElectronIsoCollection_h


#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include <vector>

namespace reco{

  typedef edm::AssociationVector<reco::GsfElectronRefProd,std::vector<double> > GsfElectronIsoCollection;
  

  typedef  GsfElectronIsoCollection::value_type     GsfElectronIso;
  typedef  edm::Ref<GsfElectronIsoCollection>       GsfElectronIsoCollectionRef;
  typedef  edm::RefProd<GsfElectronIsoCollection>   GsfElectronIsoCollectionRefProd;
  typedef  edm::RefVector<GsfElectronIsoCollection> GsfElectronIsoCollectionRefVector;

}

#endif
