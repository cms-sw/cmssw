#ifndef JetAlgorithms_ProtoJetComparisons_h
#define JetAlgorithms_ProtoJetComparisons_h
#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"

class ProtoJetPtGreater
{
 public:
  int operator()(const ProtoJet& pj1, const ProtoJet& pj2) const
  {
    return pj1.pt() > pj2.pt();
  }
};

#endif

