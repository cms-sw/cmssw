
import FWCore.ParameterSet.Config as cms

process = cms.Process("BTAG")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(3000))

#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#   ignoreTotal = cms.untracked.int32(1) # default is one
#)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:patTuple_micro.root'
)
)

# output module
#

process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.EventContent.EventContent_cff')


process.AOD1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('btag001.root')
)

process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )
process.unpackTV = cms.EDProducer('PATTrackAndVertexUnpacker',
 slimmedVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
 additionalTracks= cms.InputTag("lostTracks"),
 packedCandidates = cms.InputTag("packedPFCandidates")
)

process.ak4JetTracksAssociatorAtVertexPF.jets = cms.InputTag("slimmedJets")
process.ak4JetTracksAssociatorAtVertexPF.tracks = cms.InputTag("unpackTV")
process.impactParameterTagInfos.primaryVertex = cms.InputTag("unpackTV")
process.p = cms.Path(process.unpackTV*process.ak4JetTracksAssociatorAtVertexPF*process.impactParameterTagInfos*process.secondaryVertexTagInfos*process.combinedSecondaryVertexBJetTags)
process.endpath= cms.EndPath(process.AOD1)


