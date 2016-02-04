#ifndef DataFormats_TauReco_PFTauDiscriminatorByIsolation_h
#define DataFormats_TauReco_PFTauDiscriminatorByIsolation_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<PFTauRefProd,std::vector<int> > PFTauDiscriminatorByIsolationBase;
  
  class PFTauDiscriminatorByIsolation : public PFTauDiscriminatorByIsolationBase {
  public:
    PFTauDiscriminatorByIsolation() :
      PFTauDiscriminatorByIsolationBase()
      { }
    
    PFTauDiscriminatorByIsolation(const reco::PFTauRefProd & ref) :
      PFTauDiscriminatorByIsolationBase(ref)
      { }
    
    PFTauDiscriminatorByIsolation(const PFTauDiscriminatorByIsolationBase &v) :
      PFTauDiscriminatorByIsolationBase(v)
      { }
  };
  
  typedef PFTauDiscriminatorByIsolation::value_type PFTauDiscriminatorByIsolationVT;  
  typedef edm::Ref<PFTauDiscriminatorByIsolation> PFTauDiscriminatorByIsolationRef;  
  typedef edm::RefProd<PFTauDiscriminatorByIsolation> PFTauDiscriminatorByIsolationRefProd;  
  typedef edm::RefVector<PFTauDiscriminatorByIsolation> PFTauDiscriminatorByIsolationRefVector; 
}
#endif
