#ifndef JetProducers_CaloJetMaker_h
#define JetProducers_CaloJetMaker_h

#include "DataFormats/JetObjects/interface/CaloJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet2.h"


/// Make CaloJet from protoobjects
class CaloJetMaker {
 public:
  CaloJet makeCaloJet (const ProtoJet2& fProtojet) const;
};

#endif
