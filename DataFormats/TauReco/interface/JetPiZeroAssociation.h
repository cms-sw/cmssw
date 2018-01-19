#ifndef DataFormats_TauReco_JetPiZeroAssociation_h
#define DataFormats_TauReco_JetPiZeroAssociation_h

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/JetReco/interface/JetCollection.h"

namespace reco {
   // This base class improves the readability of the ROOT class name by hiding
   // the template crap
   typedef edm::AssociationVector<JetRefBaseProd, std::vector<std::vector<RecoTauPiZero> > >
     JetPiZeroAssociationBase;  

   class JetPiZeroAssociation : public JetPiZeroAssociationBase {
      public: 
    JetPiZeroAssociation() :
      JetPiZeroAssociationBase()
      { }
    
    JetPiZeroAssociation(const JetRefBaseProd & ref) :
      JetPiZeroAssociationBase(ref)
      { }
    
    JetPiZeroAssociation(const JetPiZeroAssociationBase &v) :
      JetPiZeroAssociationBase(v)
      { }
  };
  
  typedef JetPiZeroAssociation::value_type JetPiZeroAssociationPiZeros;  
  typedef edm::Ref<JetPiZeroAssociation> JetPiZeroAssociationRef;  
  typedef edm::RefProd<JetPiZeroAssociation> JetPiZeroAssociationRefProd;  
  typedef edm::RefVector<JetPiZeroAssociation> JetPiZeroAssociationRefVector; 
}
#endif
