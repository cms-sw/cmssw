import FWCore.ParameterSet.Config as cms

process = cms.Process("S2")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:patTuple_mini.root")
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
from RecoMET.METProducers.PFMET_cfi import pfMet

process.chs = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("fromPV"))

process.ak5PFJets = ak5PFJets.clone(src = 'packedPFCandidates', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak5PFJetsCHS = ak5PFJets.clone(src = 'chs', doAreaFastjet = True) # no idea while doArea is false by default, but it's True in RECO so we have to set it
process.ak5GenJets = ak5GenJets.clone(src = 'packedGenParticles')
process.pfMet = pfMet.clone(src = "packedPFCandidates")
process.pfMet.calculateSignificance = False # this can't be easily implemented on packed PF candidates at the moment

process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START70_V6::All'



process.patJets.addJetCharge   = False
process.patJets.addBTagInfo    = True
process.patJets.getJetMCFlavour = False
process.patJets.addAssociatedTracks = False
process.patJetPartonMatch.matched = "prunedGenParticles"
process.patJetCorrFactors.primaryVertices = "offlineSlimmedPrimaryVertices"
process.patMETs.addGenMET = False # There's no point in recalculating this, and we can't remake it since we don't have genParticles beyond |eta|=5

process.load('RecoBTag.Configuration.RecoBTag_cff')
process.load('RecoJets.Configuration.RecoJetAssociations_cff')

process.load('PhysicsTools.PatAlgos.slimming.unpackedTracksAndVertices_cfi')

process.ak5JetTracksAssociatorAtVertexPF.jets = cms.InputTag("ak5PFJetsCHS")
process.ak5JetTracksAssociatorAtVertexPF.tracks = cms.InputTag("unpackedTracksAndVertices")
process.impactParameterTagInfos.primaryVertex = cms.InputTag("unpackedTracksAndVertices")
process.combinedSecondaryVertex.trackMultiplicityMin = 1 #silly sv, uses un filtered tracks.. i.e. any pt


process.p = cms.Path(
    process.patJets + process.patMETs
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.MessageLogger.suppressWarning = cms.untracked.vstring('ecalLaserCorrFilter','manystripclus53X','toomanystripclus53X')
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.options.allowUnscheduled = cms.untracked.bool(True)

process.OUT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(['drop *','keep patJets_patJets_*_*','keep *_*_*_PAT'])
)
process.endpath= cms.EndPath(process.OUT)

