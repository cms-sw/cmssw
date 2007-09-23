#ifndef EgammaCandidates_PMGsfElectronIsoNumCollection_h
#define EgammaCandidates_PMGsfElectronIsoNumCollection_h


#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include <vector>

namespace reco{

  typedef edm::AssociationVector<reco::PixelMatchGsfElectronRefProd,std::vector<int> > PMGsfElectronIsoNumCollectionBase;
  
  class PMGsfElectronIsoNumCollection : public  PMGsfElectronIsoNumCollectionBase {
  public:
    PMGsfElectronIsoNumCollection() :
      PMGsfElectronIsoNumCollectionBase()
      { }
   PMGsfElectronIsoNumCollection(const reco::PixelMatchGsfElectronRefProd & ref) :
      PMGsfElectronIsoNumCollectionBase(ref)
      { }
   PMGsfElectronIsoNumCollection(const PMGsfElectronIsoNumCollectionBase &v) :
      PMGsfElectronIsoNumCollectionBase(v)
     { }
  };


  typedef  PMGsfElectronIsoNumCollection::value_type     PMGsfElectronIsoNum;
  typedef  edm::Ref<PMGsfElectronIsoNumCollection>       PMGsfElectronIsoNumCollectionRef;
  typedef  edm::RefProd<PMGsfElectronIsoNumCollection>   PMGsfElectronIsoNumCollectionRefProd;
  typedef  edm::RefVector<PMGsfElectronIsoNumCollection> PMGsfElectronIsoNumCollectionRefVector;

}

#endif
