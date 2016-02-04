
import FWCore.ParameterSet.Config as cms

process = cms.Process("CastorProducts")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = 
cms.untracked.vstring('file:./digiout.root')
)

process.load('RecoLocalCalo.Castor.Castor_cff')

process.MyOutputModule = cms.OutputModule("PoolOutputModule",
    #outputCommands = cms.untracked.vstring('keep recoGenParticles*_*_*_*', 'keep *_castorreco_*_*', 'keep *_Castor*Reco*_*_CastorProducts'),
    #outputCommands = cms.untracked.vstring('keep *_Castor*Reco*_*_CastorProducts','drop *_CastorFastjetReco*_*_CastorProducts','drop *_CastorTowerCandidateReco*_*_CastorProducts'),
    fileName = cms.untracked.string('recooutput.root')
)

process.producer = cms.Path(process.CastorFullReco)
process.end = cms.EndPath(process.MyOutputModule)

