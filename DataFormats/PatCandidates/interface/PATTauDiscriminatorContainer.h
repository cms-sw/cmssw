#ifndef DataFormats_PatCandidates_PATTauDiscriminatorContainer_h
#define DataFormats_PatCandidates_PATTauDiscriminatorContainer_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include <vector>

namespace pat {
  struct PATSingleTauDiscriminatorContainer {
      std::vector<float> rawValues;
      std::vector<bool> workingPoints;
      operator float() const { return rawValues.at(0); }
      
      PATSingleTauDiscriminatorContainer(){}
      PATSingleTauDiscriminatorContainer(float rawInit){ rawValues.push_back(rawInit); }
  };

  typedef edm::AssociationVector<pat::TauRefProd,std::vector<PATSingleTauDiscriminatorContainer> > PATTauDiscriminatorContainerBase;

  class PATTauDiscriminatorContainer : public PATTauDiscriminatorContainerBase {
  public:
    /// empty constructor
    PATTauDiscriminatorContainer(); // : PATTauDiscriminatorContainerBase() {}
    /// constructor from reference to pat::Tau
    PATTauDiscriminatorContainer(const pat::TauRefProd & ref) : PATTauDiscriminatorContainerBase(ref) {}
    /// constructor from base object   
    PATTauDiscriminatorContainer(const PATTauDiscriminatorContainerBase &v) : PATTauDiscriminatorContainerBase(v) {}
  };

  typedef pat::PATTauDiscriminatorContainer::value_type PATTauDiscriminatorContainerVT;  
  typedef edm::Ref<pat::PATTauDiscriminatorContainer> PATTauDiscriminatorContainerRef;  
  typedef edm::RefProd<pat::PATTauDiscriminatorContainer> PATTauDiscriminatorContainerRefProd;  
  typedef edm::RefVector<pat::PATTauDiscriminatorContainer> PATTauDiscriminatorContainerRefVector; 
}

#endif
