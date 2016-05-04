#ifndef PhysicsTools_Heppy_JetUtils_h
#define PhysicsTools_Heppy_JetUtils_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace heppy{

  struct JetUtils{

    static const pat::Jet 
    copyJet(const edm::ProductID& id, const pat::Jet& ijet, unsigned long key);

  };
};


#endif
