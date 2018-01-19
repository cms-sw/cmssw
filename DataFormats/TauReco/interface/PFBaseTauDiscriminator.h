#ifndef DataFormats_TauReco_PFBaseTauDiscriminator_h
#define DataFormats_TauReco_PFBaseTauDiscriminator_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFBaseTau.h"

#include <vector>

namespace reco {
  typedef edm::AssociationVector<PFBaseTauRefProd,std::vector<float> > PFBaseTauDiscriminatorBase;
  
  class PFBaseTauDiscriminator : public PFBaseTauDiscriminatorBase {
  public:
    PFBaseTauDiscriminator() :
      PFBaseTauDiscriminatorBase()
      { }
    
    PFBaseTauDiscriminator(const reco::PFBaseTauRefProd & ref) :
      PFBaseTauDiscriminatorBase(ref)
      { }
    
    PFBaseTauDiscriminator(const PFBaseTauDiscriminatorBase &v) :
      PFBaseTauDiscriminatorBase(v)
      { }
  };
  
  typedef PFBaseTauDiscriminator::value_type PFBaseTauDiscriminatorVT;  
  typedef edm::Ref<PFBaseTauDiscriminator> PFBaseTauDiscriminatorRef;  
  typedef edm::RefProd<PFBaseTauDiscriminator> PFBaseTauDiscriminatorRefProd;  
  typedef edm::RefVector<PFBaseTauDiscriminator> PFBaseTauDiscriminatorRefVector; 
}
#endif
