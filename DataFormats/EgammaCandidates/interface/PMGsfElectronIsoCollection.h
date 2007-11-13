#ifndef EgammaCandidates_PMGsfElectronIsoCollection_h
#define EgammaCandidates_PMGsfElectronIsoCollection_h


#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include <vector>

namespace reco{

  typedef edm::AssociationVector<reco::PixelMatchGsfElectronRefProd,std::vector<double> > PMGsfElectronIsoCollection;
  

  typedef  PMGsfElectronIsoCollection::value_type     PMGsfElectronIso;
  typedef  edm::Ref<PMGsfElectronIsoCollection>       PMGsfElectronIsoCollectionRef;
  typedef  edm::RefProd<PMGsfElectronIsoCollection>   PMGsfElectronIsoCollectionRefProd;
  typedef  edm::RefVector<PMGsfElectronIsoCollection> PMGsfElectronIsoCollectionRefVector;

}

#endif
