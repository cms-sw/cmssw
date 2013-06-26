#ifndef JetProducers_JetMaker_h
#define JetProducers_JetMaker_h

/// Algorithm to make jet flavor specific information from transient protojet
/// Author: F.Ratnikov, UMd
/// Mar. 8, 2006
/// $Id: JetMaker.h,v 1.1 2009/08/24 14:35:59 srappocc Exp $

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"

class CaloSubdetectorGeometry;


/// Make Jets from protoobjects
namespace JetMaker {
  /// Make CaloJet specifics. Assumes ProtoJet is made from CaloTowerCandidates
  bool makeSpecific (const JetReco::InputCollection& fConstituents, 
		     const CaloSubdetectorGeometry& fTowerGeometry,
		     reco::CaloJet::Specific* fJetSpecific);
  
  /// Make PFlowJet specifics. Assumes ProtoJet is made from ParticleFlowCandidates
  bool makeSpecific (const JetReco::InputCollection& fConstituents, 
		     reco::PFJet::Specific* fJetSpecific);

  /// Make GenJet specifics. Assumes ProtoJet is made from HepMCCandidate
  bool makeSpecific (const JetReco::InputCollection& fConstituents, 
		     reco::GenJet::Specific* fJetSpecific);

  /// converts eta to the corresponding HCAL subdetector.
  HcalSubdetector hcalSubdetector (int fEta);
}

#endif
