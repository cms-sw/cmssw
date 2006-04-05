#ifndef JetProducers_JetMaker_h
#define JetProducers_JetMaker_h

/// Algorithm to convert transient protojets into persistent jets
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id: JetMaker.h,v 1.2 2006/03/31 20:57:51 fedor Exp $

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"


/// Make Jets from protoobjects
class JetMaker {
 public:
  /// Verify if ProtoJet may be converted to CaloJet
  bool convertableToCaloJet (const ProtoJet& fProtojet) const;
  /// Verify if ProtoJet may be converted to GenJet
  bool convertableToGenJet (const ProtoJet& fProtojet) const;
  /// Make CaloJet. Assumes ProtoJet is made from CaloTowerCandidates
  CaloJet makeCaloJet (const ProtoJet& fProtojet) const;
  /// Make GenJet. Assumes ProtoJet is made from HepMCCandidate
  GenJet makeGenJet (const ProtoJet& fProtojet) const;
};

#endif
