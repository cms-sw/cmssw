import FWCore.ParameterSet.Config as cms

vbfGenJetFilterA = cms.EDFilter("VBFGenJetFilter",

  inputTag_GenJetCollection = cms.untracked.InputTag('ak4GenJetsNoNu'),

  oppositeHemisphere = cms.untracked.bool  ( False), # Require j1_eta*j2_eta<0
  minPt              = cms.untracked.double(    40), # Minimum dijet jet_pt
  minEta             = cms.untracked.double(  -4.8), # Minimum dijet jet_eta
  maxEta             = cms.untracked.double(   4.8), # Maximum dijet jet_eta
  minInvMass         = cms.untracked.double( 1000.), # Minimum dijet invariant mass
  maxInvMass         = cms.untracked.double(99999.), # Maximum dijet invariant mass
  minDeltaPhi        = cms.untracked.double(  -1.0), # Minimum dijet delta phi
  maxDeltaPhi        = cms.untracked.double(  2.15), # Maximum dijet delta phi
  minDeltaEta        = cms.untracked.double(   3.0), # Minimum dijet delta eta
  maxDeltaEta        = cms.untracked.double(99999.)  # Maximum dijet delta eta

)

vbfGenJetFilterB = cms.EDFilter("VBFGenJetFilter",

  inputTag_GenJetCollection = cms.untracked.InputTag('ak4GenJetsNoNu'),

  oppositeHemisphere = cms.untracked.bool  ( False), # Require j1_eta*j2_eta<0
  minPt              = cms.untracked.double(    40), # Minimum dijet jet_pt
  minEta             = cms.untracked.double(  -4.8), # Minimum dijet jet_eta
  maxEta             = cms.untracked.double(   4.8), # Maximum dijet jet_eta
  minInvMass         = cms.untracked.double( 1000.), # Minimum dijet invariant mass
  maxInvMass         = cms.untracked.double(99999.), # Maximum dijet invariant mass
  minDeltaPhi        = cms.untracked.double(  2.15), # Minimum dijet delta phi
  maxDeltaPhi        = cms.untracked.double(   3.2), # Maximum dijet delta phi
  minDeltaEta        = cms.untracked.double(   3.0), # Minimum dijet delta eta
  maxDeltaEta        = cms.untracked.double(99999.)  # Maximum dijet delta eta

)

vbfGenJetFilterC = cms.EDFilter("VBFGenJetFilter",

  inputTag_GenJetCollection = cms.untracked.InputTag('ak4GenJetsNoNu'),

  oppositeHemisphere = cms.untracked.bool  ( False), # Require j1_eta*j2_eta<0
  minPt              = cms.untracked.double(    40), # Minimum dijet jet_pt
  minEta             = cms.untracked.double(  -4.8), # Minimum dijet jet_eta
  maxEta             = cms.untracked.double(   4.8), # Maximum dijet jet_eta
  minInvMass         = cms.untracked.double( 1000.), # Minimum dijet invariant mass
  maxInvMass         = cms.untracked.double(99999.), # Maximum dijet invariant mass
  minDeltaPhi        = cms.untracked.double(  -1.0), # Minimum dijet delta phi
  maxDeltaPhi        = cms.untracked.double(   3.2), # Maximum dijet delta phi
  minDeltaEta        = cms.untracked.double(   3.0), # Minimum dijet delta eta
  maxDeltaEta        = cms.untracked.double(99999.)  # Maximum dijet delta eta

)