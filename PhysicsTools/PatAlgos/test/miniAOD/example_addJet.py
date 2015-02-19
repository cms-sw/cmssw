import FWCore.ParameterSet.Config as cms

process = cms.Process("S2")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:patTuple_mini.root")
)
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarPileUpMINIAODSIM
process.source.fileNames = filesRelValProdTTbarPileUpMINIAODSIM

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets

process.chs = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("fromPV"))

process.ak4PFJets = ak4PFJets.clone(src = 'packedPFCandidates', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak4PFJetsCHS = ak4PFJets.clone(src = 'chs', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak4GenJets = ak4GenJets.clone(src = 'packedGenParticles')

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
addJetCollection(
   process,
   postfix   = "",
   labelName = 'AK4PFCHS',
   jetSource = cms.InputTag('ak4PFJetsCHS'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   pfCandidates = cms.InputTag('packedPFCandidates'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   btagDiscriminators = [ 'pfCombinedSecondaryVertexBJetTags', 'pfCombinedInclusiveSecondaryVertexV2BJetTags' ],
   genJetCollection=cms.InputTag('ak4GenJets'),
   genParticles=cms.InputTag('prunedGenParticles')
   )
# if using legacy jet flavour (not used by default)
process.patJetPartonsLegacy.skipFirstN = cms.uint32(0) # do not skip first 6 particles, we already pruned some!
process.patJetPartonsLegacy.acceptNoDaughters = cms.bool(True) # as we drop intermediate stuff, we need to accept quarks with no siblings


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.options.allowUnscheduled = cms.untracked.bool(True)

process.OUT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(['drop *','keep patJets_patJetsAK4PFCHS_*_*','keep *_*_*_PAT'])
)
process.endpath= cms.EndPath(process.OUT)

