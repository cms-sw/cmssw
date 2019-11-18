#ifndef DataFormats_TauReco_PFTauDiscriminatorContainer_h
#define DataFormats_TauReco_PFTauDiscriminatorContainer_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include <vector>

namespace reco {
  struct PFSingleTauDiscriminatorContainer {
      std::vector<float> rawValues;
      std::vector<bool> workingPoints;
      operator float() const { return rawValues.at(0); }
      
      PFSingleTauDiscriminatorContainer(){}
      PFSingleTauDiscriminatorContainer(float rawInit){ rawValues.push_back(rawInit); }
  };

  typedef edm::AssociationVector<PFTauRefProd,std::vector<PFSingleTauDiscriminatorContainer> > PFTauDiscriminatorContainerBase;
  
  class PFTauDiscriminatorContainer : public PFTauDiscriminatorContainerBase {
  public:
    PFTauDiscriminatorContainer() :
      PFTauDiscriminatorContainerBase()
      { }
    
    PFTauDiscriminatorContainer(const reco::PFTauRefProd & ref) :
      PFTauDiscriminatorContainerBase(ref)
      { }
    
    PFTauDiscriminatorContainer(const PFTauDiscriminatorContainerBase &v) :
      PFTauDiscriminatorContainerBase(v)
      { }
  };
  
  typedef PFTauDiscriminatorContainer::value_type PFTauDiscriminatorContainerVT;  
  typedef edm::Ref<PFTauDiscriminatorContainer> PFTauDiscriminatorContainerRef;  
  typedef edm::RefProd<PFTauDiscriminatorContainer> PFTauDiscriminatorContainerRefProd;  
  typedef edm::RefVector<PFTauDiscriminatorContainer> PFTauDiscriminatorContainerRefVector; 
}
#endif
