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
'/store/data/CRAFT09/AlCaPhiSymHcal/RAW/v1/000/112/257/56E76AD5-9B92-DE11-9A36-0030487A1990.root'
))

process.load("HLTriggerOffline.special.hltHPSdqm_cfi")
process.hltHPSdqm.SaveToRootFile=cms.bool(True)

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)
 
process.dqmOut = cms.OutputModule("PoolOutputModule",
      fileName = cms.untracked.string('dqmHltHPS.root'),
      outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
  )
 
process.p = cms.Path(process.hltHPSdqm+process.MEtoEDMConverter)
 
process.ep=cms.EndPath(process.dqmOut) 
