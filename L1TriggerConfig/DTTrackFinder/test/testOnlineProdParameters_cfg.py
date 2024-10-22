import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTTFParametersOnline_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
                             toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1MuDTTFParametersRcd'),
    data = cms.vstring('L1MuDTTFParameters')
    )),
                             verbose = cms.untracked.bool(True)
                             )

process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
            record = cms.string('L1MuDTTFParametersRcd'),
                    type = cms.string('L1MuDTTFParameters'),
                    key = cms.string('GlobalRun_Cosmics_Nov08_MB1')
            ))


process.p = cms.Path(process.get)
