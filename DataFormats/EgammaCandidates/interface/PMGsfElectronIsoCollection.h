#ifndef EgammaCandidates_PMGsfElectronIsoCollection_h
#define EgammaCandidates_PMGsfElectronIsoCollection_h


#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include <vector>

namespace reco{

  typedef edm::AssociationVector<reco::PixelMatchGsfElectronRefProd,std::vector<double> > PMGsfElectronIsoCollectionBase;
  
  class PMGsfElectronIsoCollection : public  PMGsfElectronIsoCollectionBase {
  public:
    PMGsfElectronIsoCollection() :
      PMGsfElectronIsoCollectionBase()
      { }
   PMGsfElectronIsoCollection(const reco::PixelMatchGsfElectronRefProd & ref) :
      PMGsfElectronIsoCollectionBase(ref)
      { }
   PMGsfElectronIsoCollection(const PMGsfElectronIsoCollectionBase &v) :
      PMGsfElectronIsoCollectionBase(v)
     { }
  };


  typedef  PMGsfElectronIsoCollection::value_type     PMGsfElectronIso;
  typedef  edm::Ref<PMGsfElectronIsoCollection>       PMGsfElectronIsoCollectionRef;
  typedef  edm::RefProd<PMGsfElectronIsoCollection>   PMGsfElectronIsoCollectionRefProd;
  typedef  edm::RefVector<PMGsfElectronIsoCollection> PMGsfElectronIsoCollectionRefVector;

}

#endif
