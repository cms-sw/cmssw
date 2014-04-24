
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
process.load('Configuration/StandardSequences/Geometry_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/EventContent/EventContent_cff')


process.AOD1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('btag001.root')
)

process.GlobalTag.globaltag = 'POSTLS162_V1::All'

process.unpackTV = cms.EDProducer('PATTrackAndVertexUnpacker',
 slimmedVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
 additionalTracks= cms.InputTag("lostTracks"),
 packedCandidates = cms.InputTag("packedPFCandidates")
)

process.ak5JetTracksAssociatorAtVertexPF.jets = cms.InputTag("slimmedJets")
process.ak5JetTracksAssociatorAtVertexPF.tracks = cms.InputTag("unpackTV")
process.impactParameterTagInfos.primaryVertex = cms.InputTag("unpackTV")
process.combinedSecondaryVertex.trackMultiplicityMin = 1 #silly sv, uses un filtered tracks.. i.e. any pt
process.p = cms.Path(process.unpackTV*process.ak5JetTracksAssociatorAtVertexPF*process.impactParameterTagInfos*process.secondaryVertexTagInfos*process.combinedSecondaryVertexBJetTags)
process.endpath= cms.EndPath(process.AOD1)


