
import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorProducts")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring('file:digiout.root')
)

process.load('RecoLocalCalo.Castor.Castor_cff')

process.MyOutputModule = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_Castor*Reco_*_*'),
    fileName = cms.untracked.string('recooutput.root')
)

process.producer = cms.Path(process.CastorFullReco)
process.end = cms.EndPath(process.MyOutputModule)

