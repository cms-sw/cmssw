#ifndef DataFormats_PatCandidates_PATTauDiscriminator_h
#define DataFormats_PatCandidates_PATTauDiscriminator_h
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include <vector>

namespace pat {
  typedef edm::AssociationVector<pat::TauRefProd,std::vector<float> > PATTauDiscriminatorBase;

  class PATTauDiscriminator : public PATTauDiscriminatorBase {
  public:
    /// empty constructor
    PATTauDiscriminator(); // : PATTauDiscriminatorBase() {}
    /// constructor from reference to pat::Tau
    PATTauDiscriminator(const pat::TauRefProd & ref) : PATTauDiscriminatorBase(ref) {}
    /// constructor from base object   
    PATTauDiscriminator(const PATTauDiscriminatorBase &v) : PATTauDiscriminatorBase(v) {}
  };

  typedef pat::PATTauDiscriminator::value_type PATTauDiscriminatorVT;  
  typedef edm::Ref<pat::PATTauDiscriminator> PATTauDiscriminatorRef;  
  typedef edm::RefProd<pat::PATTauDiscriminator> PATTauDiscriminatorRefProd;  
  typedef edm::RefVector<pat::PATTauDiscriminator> PATTauDiscriminatorRefVector; 
}

#endif
