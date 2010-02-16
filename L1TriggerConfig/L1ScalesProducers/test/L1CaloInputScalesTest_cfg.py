
import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
# "old-style" ECAL scale
process.load("CalibCalorimetry.EcalTPGTools.ecalTPGScale_cff")

# "old-style" HCAL scale
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

#include "L1TriggerConfig/RCTConfigProducers/data/L1RCTConfig.cff"
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#keep the logging output to a nice level
process.MessageLogger = cms.Service("MessageLogger")

process.test = cms.EDAnalyzer("L1CaloInputScaleTester")

process.p = cms.Path(process.test)


