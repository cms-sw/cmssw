#ifndef BTauReco_JetEisolAssociation_h
#define BTauReco_JetEisolAssociation_h
// \class JetEisolAssociation
// 
// \short association of tracks to jet (was JetWithEisol)
// 
//

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include <vector>

namespace reco {
  typedef
  edm::AssociationMap<edm::OneToValue<CaloJetCollection, float, unsigned int> >
                      JetEisolAssociationCollection;
  
  typedef
  JetEisolAssociationCollection::value_type JetEisolAssociation;
  
  typedef
  edm::Ref<JetEisolAssociationCollection> JetEisolAssociationRef;
  
  typedef
  edm::RefProd<JetEisolAssociationCollection> JetEisolAssociationRefProd;
  
  typedef
  edm::RefVector<JetEisolAssociationCollection> JetEisolAssociationRefVector; 
}
#endif
