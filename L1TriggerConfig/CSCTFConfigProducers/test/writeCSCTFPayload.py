import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigOnline_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCPtLutConfigOnline_cfi") 

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.source = cms.Source("EmptySource")

from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = 'sqlite_file:csctf.db')
process.L1CondDBPayloadWriter.writeL1TriggerKey = cms.bool(False)

process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
     record = cms.string('L1MuCSCTFConfigurationRcd'),  
     type = cms.string('L1MuCSCTFConfiguration'),
     key = cms.string('200710') # <-- you may want to change it depending on the payload
     ))

## process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
##    record = cms.string('L1MuCSCPtLutRcd'),  
##    type = cms.string('L1MuCSCPtLut'),
##    key = cms.string('4') # <-- you may want to change it depending on the payload
##                 ))

process.p = cms.Path(process.L1CondDBPayloadWriter)
