import FWCore.ParameterSet.Config as cms

slimmedJets = cms.EDProducer("PATJetSlimmer",
   src = cms.InputTag("selectedPatJets"),
   packedPFCandidates = cms.InputTag("packedPFCandidates"),
   dropJetVars = cms.string("1"),
   dropDaughters = cms.string("0"),
   rekeyDaughters = cms.string("1"),
   dropTrackRefs = cms.string("1"),
   dropSpecific = cms.string("0"),
   dropTagInfos = cms.string("1"),
   modifyJets = cms.bool(True),
   mixedDaughters = cms.bool(False),
   modifierConfig = cms.PSet( modifications = cms.VPSet() )
)
slimmedJetsAK8 = cms.EDProducer("PATJetSlimmer",
   src = cms.InputTag("packedPatJetsAK8"),
   packedPFCandidates = cms.InputTag("packedPFCandidates"),
   dropJetVars = cms.string("1"),
   dropDaughters = cms.string("0"),
   rekeyDaughters = cms.string("1"),
   dropTrackRefs = cms.string("1"),
   dropSpecific = cms.string("0"),
   dropTagInfos = cms.string("0"),
   modifyJets = cms.bool(True),
   mixedDaughters = cms.bool(False),
   modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

