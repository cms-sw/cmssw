import FWCore.ParameterSet.Config as cms

iPSet = cms.PSet(
    coneSize = cms.double(0.3),
    isolateAgainst = cms.string('h+'),
    isolationAlgo = cms.string('ElectronPFIsolationWithMapBasedVeto'),
    miniAODVertexCodes = cms.vuint32(2, 3),
    particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp","gedGsfElectronsTmp"),
    vertexIndex = cms.int32(0)
)