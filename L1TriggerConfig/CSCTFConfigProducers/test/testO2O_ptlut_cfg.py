# to test the communication with DBS and produce the ptlut 
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')


# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCPtLutConfigOnline_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process )

process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1MuCSCPtLutRcd'),
   data = cms.vstring('L1MuCSCPtLut')
   )),
   verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.getter)



process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
        record = cms.string('L1MuCSCPtLutRcd'),
        type = cms.string('L1MuCSCPtLut'),
        key = cms.string('3')
))

