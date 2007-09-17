#ifndef DataFormats_TauReco_CaloTauDiscriminatorByIsolation_h
#define DataFormats_TauReco_CaloTauDiscriminatorByIsolation_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<CaloTauRefProd,std::vector<int> > CaloTauDiscriminatorByIsolationBase;
  
  class CaloTauDiscriminatorByIsolation : public CaloTauDiscriminatorByIsolationBase {
  public:
    CaloTauDiscriminatorByIsolation() :
      CaloTauDiscriminatorByIsolationBase()
      { }
    
    CaloTauDiscriminatorByIsolation(const reco::CaloTauRefProd & ref) :
      CaloTauDiscriminatorByIsolationBase(ref)
      { }
    
    CaloTauDiscriminatorByIsolation(const CaloTauDiscriminatorByIsolationBase &v) :
      CaloTauDiscriminatorByIsolationBase(v)
      { }
  };
  
  typedef CaloTauDiscriminatorByIsolation::value_type CaloTauDiscriminatorByIsolationVT;  
  typedef edm::Ref<CaloTauDiscriminatorByIsolation> CaloTauDiscriminatorByIsolationRef;  
  typedef edm::RefProd<CaloTauDiscriminatorByIsolation> CaloTauDiscriminatorByIsolationRefProd;  
  typedef edm::RefVector<CaloTauDiscriminatorByIsolation> CaloTauDiscriminatorByIsolationRefVector; 
}
#endif
