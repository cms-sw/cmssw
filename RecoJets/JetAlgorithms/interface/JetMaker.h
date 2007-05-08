#ifndef JetProducers_JetMaker_h
#define JetProducers_JetMaker_h

/// Algorithm to convert transient protojets into persistent jets
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id: JetMaker.h,v 1.8 2007/05/04 23:23:04 fedor Exp $

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/GenericJet.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class CaloSubdetectorGeometry;


/// Make Jets from protoobjects
class JetMaker {
 public:
  reco::BasicJet makeBasicJet (const ProtoJet& fProtojet) const;
  /// Make CaloJet. Assumes ProtoJet is made from CaloTowerCandidates
  reco::CaloJet makeCaloJet (const ProtoJet& fProtojet, const CaloSubdetectorGeometry& fTowerGeometry) const;
  /// Make PFlowJet. Assumes ProtoJet is made from ParticleFlowCandidates
  reco::PFJet makePFJet (const ProtoJet& fProtojet) const;
  /// Make GenJet. Assumes ProtoJet is made from HepMCCandidate
  reco::GenJet makeGenJet (const ProtoJet& fProtojet) const;
  /// Generic jet: assumes nothing
  reco::GenericJet makeGenericJet (const ProtoJet& fProtojet) const;

  /// converts eta to the corresponding HCAL subdetector.
  static HcalSubdetector hcalSubdetector (int fEta);
};

#endif
