import FWCore.ParameterSet.Config as cms

process = cms.Process("S2")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:patTuple_mini.root")
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets

process.chs = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("fromPV"))

process.ak5PFJets = ak5PFJets.clone(src = 'packedPFCandidates', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak5PFJetsCHS = ak5PFJets.clone(src = 'chs', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak5GenJets = ak5GenJets.clone(src = 'packedGenParticles')

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START70_V6::All'

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
addJetCollection(
   process,
   postfix   = "",
   labelName = 'AK5PFCHS',
   jetSource = cms.InputTag('ak5PFJetsCHS'),
   trackSource = cms.InputTag('unpackedTracksAndVertices'), 	
   pvSource = cms.InputTag('unpackedTracksAndVertices'), 
   jetCorrections = ('AK5PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute']), 'Type-2'),
   btagDiscriminators = [      'combinedSecondaryVertexBJetTags'     ]
   )
process.patJetPartonMatchPatJetsAK5PFCHS.matched = "prunedGenParticles"
process.patJetPartons.src = "prunedGenParticles"
process.patJetPartons.skipFirstN = cms.uint32(0) # do not skip first 6 particles, we already pruned some!
process.patJetPartons.acceptNoDaughters = cms.bool(True) # as we drop intermediate stuff, we need to accept quarks with no siblings
process.patJetCorrFactorsPatJetsAK5PFCHS.primaryVertices = "offlineSlimmedPrimaryVertices"

#recreate tracks and pv for btagging
process.load('PhysicsTools.PatAlgos.slimming.unpackedTracksAndVertices_cfi')
process.combinedSecondaryVertex.trackMultiplicityMin = 1 #silly sv, uses un filtered tracks.. i.e. any pt

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.options.allowUnscheduled = cms.untracked.bool(True)

process.OUT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(['drop *','keep patJets_patJetsAK5PFCHS_*_*','keep *_*_*_PAT'])
)
process.endpath= cms.EndPath(process.OUT)

