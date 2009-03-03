import FWCore.ParameterSet.Config as cms

process = cms.Process("HITval")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20000) )

process.source = cms.Source("PoolSource",
    # put here 10 TeV RelVal minbias sample with HLTDEBUG, RAW & RECO
    fileNames = cms.untracked.vstring(
'rfio:/castor/cern.ch/user/s/safronov/codeval/HLTFromPureRaw_310_2.root'
)
)

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

######### select L1 menu for which you want to validate/evaluate rates
# in L1TriggerConfig.L1GtConfigProducers.Luminosity
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_cff")
#process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

process.load("HLTriggerOffline.special.hltHITval_cfi")
process.hltHITval.doL1Prescaling=cms.bool(True)
process.hltHITval.produceRates=cms.bool(True)
process.hltHITval.luminosity=cms.double(8e29)
process.hltHITval.sampleCrossSection=cms.double(7.53E10)
process.hltHITval.hltL3FilterLabel=cms.InputTag("hltIsolPixelTrackFilter::HLT1")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.MEtoEDMConverter.verbose = cms.untracked.int32(1)

process.dqmOut = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('dqmHltHITval_minBias.root'),
     outputCommands = cms.untracked.vstring("drop *", "keep *_MEtoEDMConverter_*_*")
 )

process.p = cms.Path(process.hltHITval+process.MEtoEDMConverter)

process.ep=cms.EndPath(process.dqmOut)
