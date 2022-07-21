import FWCore.ParameterSet.Config as cms

def customiseHeavyFlavorDQMForMiniAOD(process):
    if hasattr(process, "bphWriteSpecificDecayForDQM"):
        process.bphWriteSpecificDecayForDQM.pVertexLabel = cms.string('offlineSlimmedPrimaryVertices')
        
        process.bphWriteSpecificDecayForDQM.pcCandsLabel = cms.string('packedPFCandidates')
        process.bphWriteSpecificDecayForDQM.pfCandsLabel = cms.string('')
        
        process.bphWriteSpecificDecayForDQM.patMuonLabel = cms.string('slimmedMuons')
        
        process.bphWriteSpecificDecayForDQM.kSCandsLabel = cms.string('slimmedKshortVertices')
        process.bphWriteSpecificDecayForDQM.k0CandsLabel = cms.string('')
        
        process.bphWriteSpecificDecayForDQM.lSCandsLabel = cms.string('slimmedLambdaVertices')
        process.bphWriteSpecificDecayForDQM.l0CandsLabel = cms.string('')
    
    if hasattr(process, "heavyFlavorDQM"):
        process.heavyFlavorDQM.pvCollection = cms.InputTag('offlineSlimmedPrimaryVertices')
    
    return process
