#ifndef DataFormats_TauReco_PFJetChargedHadronAssociation_h
#define DataFormats_TauReco_PFJetChargedHadronAssociation_h

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

namespace reco 
{
  // This base class improves the readability of the ROOT class name by hiding
  // the template crap
  typedef edm::AssociationVector<PFJetRefProd, std::vector<std::vector<PFRecoTauChargedHadron> > > PFJetChargedHadronAssociationBase;  

  class PFJetChargedHadronAssociation : public PFJetChargedHadronAssociationBase 
  {
   public: 
    PFJetChargedHadronAssociation() 
      : PFJetChargedHadronAssociationBase()
    {}
    
    PFJetChargedHadronAssociation(const reco::PFJetRefProd& ref) 
      : PFJetChargedHadronAssociationBase(ref)
    {}
    
    PFJetChargedHadronAssociation(const PFJetChargedHadronAssociationBase& v) 
      : PFJetChargedHadronAssociationBase(v)
    {}
  };
  
  typedef PFJetChargedHadronAssociation::value_type PFJetChargedHadronAssociationChHadrons;  
  typedef edm::Ref<PFJetChargedHadronAssociation> PFJetChargedHadronAssociationRef;  
  typedef edm::RefProd<PFJetChargedHadronAssociation> PFJetChargedHadronAssociationRefProd;  
  typedef edm::RefVector<PFJetChargedHadronAssociation> PFJetChargedHadronAssociationRefVector; 
}
#endif
