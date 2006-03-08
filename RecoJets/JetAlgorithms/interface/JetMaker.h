#ifndef JetProducers_JetMaker_h
#define JetProducers_JetMaker_h

/// Algorithm to convert transient protojets into persistent jets
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id$

#include "DataFormats/JetObjects/interface/CaloJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"


/// Make Jets from protoobjects
class JetMaker {
 public:
  /// Make CaloJet. Assumes ProtoJet is made from CaloTowerCandidates
  CaloJet makeCaloJet (const ProtoJet& fProtojet) const;
};

#endif
