import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

#process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger", 
     destinations = cms.untracked.vstring('O2OValidation') 
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("OnlineDB.SiStripO2O.SiStripO2OValidationParameters_cfi")
process.o2ovalidation = cms.EDAnalyzer('SiStripO2OValidation',
                              process.SiStripO2OValidationParameters
                              )
process.o2ovalidation.ValidateNoise = True
process.o2ovalidation.ValidateFEDCabling = True
process.o2ovalidation.ValidatePedestal = True
process.o2ovalidation.ValidateQuality = True
process.o2ovalidation.ValidateThreshold = True
process.o2ovalidation.ValidateAPVTiming = True
process.o2ovalidation.ValidateAPVLatency = True
process.o2ovalidation.RootFile ="SiStripO2OValidation.root"
                              

process.p = cms.Path(process.o2ovalidation)
