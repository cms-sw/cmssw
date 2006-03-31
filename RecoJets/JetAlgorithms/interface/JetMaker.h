#ifndef JetProducers_JetMaker_h
#define JetProducers_JetMaker_h

/// Algorithm to convert transient protojets into persistent jets
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id: JetMaker.h,v 1.1 2006/03/08 20:30:29 fedor Exp $

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"


/// Make Jets from protoobjects
class JetMaker {
 public:
  /// Make CaloJet. Assumes ProtoJet is made from CaloTowerCandidates
  CaloJet makeCaloJet (const ProtoJet& fProtojet) const;
};

#endif
