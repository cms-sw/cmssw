#ifndef BTauReco_JetCrystalsAssociation_h
#define BTauReco_JetCrystalsAssociation_h
// \class JetCrystalsAssociation
// 
// \short association of Ecal Crystals to jet 
// 
//

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include <vector>

namespace reco {
  typedef math::PtEtaPhiELorentzVector           EMLorentzVector;
  typedef math::PtEtaPhiELorentzVectorCollection EMLorentzVectorCollection;
  typedef math::PtEtaPhiELorentzVectorRef        EMLorentzVectorRef;
  typedef math::PtEtaPhiELorentzVectorRefProd    EMLorentzVectorRefProd;
  typedef math::PtEtaPhiELorentzVectorRefVector  EMLorentzVectorRefVector;

  /* silly workaround for a bug in ROOT's dictionaries:
   * registering for autoload a class whose name is longer than 1035 characters causes a buffer overflow 
  typedef
    std::vector<std::pair<edm::RefToBase<Jet>, EMLorentzVectorRefVector>  >
    JetCrystalsAssociationCollection;
  */
        
  struct JetCrystalsAssociationCollection : 
    public std::vector<std::pair<edm::RefToBase<Jet>, EMLorentzVectorRefVector> >
  {
    typedef std::vector<std::pair<edm::RefToBase<Jet>, EMLorentzVectorRefVector> > base_class;

    JetCrystalsAssociationCollection() :
      base_class() { }
    
    JetCrystalsAssociationCollection(size_type n) :
      base_class(n) { }
    
    JetCrystalsAssociationCollection(size_type n, const value_type & t) :
      base_class(n, t) { }
    
    JetCrystalsAssociationCollection(const base_class & v) :
      base_class(v) { }

    template <class InputIterator>
    JetCrystalsAssociationCollection(InputIterator first, InputIterator last) :
      base_class(first, last) { }
  };
    
  typedef
  JetCrystalsAssociationCollection::value_type JetCrystalsAssociation;
  
  typedef
  edm::Ref<JetCrystalsAssociationCollection> JetCrystalsAssociationRef;
  
  typedef
  edm::RefProd<JetCrystalsAssociationCollection> JetCrystalsAssociationRefProd;
  
  typedef
  edm::RefVector<JetCrystalsAssociationCollection> JetCrystalsAssociationRefVector; 
}
#endif
