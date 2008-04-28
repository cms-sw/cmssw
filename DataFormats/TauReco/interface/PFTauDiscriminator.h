#ifndef DataFormats_TauReco_PFTauDiscriminator_h
#define DataFormats_TauReco_PFTauDiscriminator_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<PFTauRefProd,std::vector<float> > PFTauDiscriminatorBase;
  
  class PFTauDiscriminator : public PFTauDiscriminatorBase {
  public:
    PFTauDiscriminator() :
      PFTauDiscriminatorBase()
      { }
    
    PFTauDiscriminator(const reco::PFTauRefProd & ref) :
      PFTauDiscriminatorBase(ref)
      { }
    
    PFTauDiscriminator(const PFTauDiscriminatorBase &v) :
      PFTauDiscriminatorBase(v)
      { }
  };
  
  typedef PFTauDiscriminator::value_type PFTauDiscriminatorVT;  
  typedef edm::Ref<PFTauDiscriminator> PFTauDiscriminatorRef;  
  typedef edm::RefProd<PFTauDiscriminator> PFTauDiscriminatorRefProd;  
  typedef edm::RefVector<PFTauDiscriminator> PFTauDiscriminatorRefVector; 
}
#endif
