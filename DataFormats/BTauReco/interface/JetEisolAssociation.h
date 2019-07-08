#ifndef BTauReco_JetEisolAssociation_h
#define BTauReco_JetEisolAssociation_h
// \class JetEisolAssociation
//
// \short association of tracks to jet (was JetWithEisol)
//
//

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

namespace reco {
  typedef std::vector<std::pair<edm::RefToBase<Jet>, float> > JetEisolAssociationCollection;

  typedef JetEisolAssociationCollection::value_type JetEisolAssociation;

  typedef edm::Ref<JetEisolAssociationCollection> JetEisolAssociationRef;

  typedef edm::FwdRef<JetEisolAssociationCollection> JetEisolAssociationFwdRef;

  typedef edm::RefProd<JetEisolAssociationCollection> JetEisolAssociationRefProd;

  typedef edm::RefVector<JetEisolAssociationCollection> JetEisolAssociationRefVector;
}  // namespace reco
#endif
