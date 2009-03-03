import FWCore.ParameterSet.Config as cms

process = cms.Process("HITdqm")
 
process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#RelVal RAW-HLTDEBUG file, or:
'rfio:/castor/cern.ch/user/s/safronov/codeval/HLTFromPureRaw_310_2.root'
))

process.load("HLTriggerOffline.special.hltHITdqm_cfi")
process.hltHITdqm.hltProcessName=cms.string("HLT1")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)
 
process.dqmOut = cms.OutputModule("PoolOutputModule",
      fileName = cms.untracked.string('dqmHltHIT.root'),
      outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
  )
 
process.p = cms.Path(process.hltHITdqm+process.MEtoEDMConverter)
 
process.ep=cms.EndPath(process.dqmOut) 
