#ifndef DataFormats_TauReco_CaloTauDiscriminator_h
#define DataFormats_TauReco_CaloTauDiscriminator_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<CaloTauRefProd,std::vector<float> > CaloTauDiscriminatorBase;
  
  class CaloTauDiscriminator : public CaloTauDiscriminatorBase {
  public:
    CaloTauDiscriminator() :
      CaloTauDiscriminatorBase()
      { }
    
    CaloTauDiscriminator(const reco::CaloTauRefProd & ref) :
      CaloTauDiscriminatorBase(ref)
      { }
    
    CaloTauDiscriminator(const CaloTauDiscriminatorBase &v) :
      CaloTauDiscriminatorBase(v)
      { }
  };
  
  typedef CaloTauDiscriminator::value_type CaloTauDiscriminatorVT;  
  typedef edm::Ref<CaloTauDiscriminator> CaloTauDiscriminatorRef;  
  typedef edm::RefProd<CaloTauDiscriminator> CaloTauDiscriminatorRefProd;  
  typedef edm::RefVector<CaloTauDiscriminator> CaloTauDiscriminatorRefVector; 
}
#endif
