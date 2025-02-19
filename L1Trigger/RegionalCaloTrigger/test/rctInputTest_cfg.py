# The following comments couldn't be translated into the new config version:

#	untracked PSet maxEvents = {untracked int32 input = 2}
#include "Configuration/ReleaseValidation/data/Services.cff"
#    include "Configuration/StandardSequences/data/FakeConditions.cff"
#    untracked PSet options = {
#        include "FWCore/Framework/test/cmsExceptionsFatalOption.cff"
#        untracked bool makeTriggerResults = true
#    }

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# 
#  ecal trig prim producer 
# # ecal tpg params
# es_module = EcalTrigPrimESProducer {
# untracked string DatabaseFile = "TPG.txt"
# #untracked string DatabaseFile = "TPG_RCT_internal.txt"
# }
# 
process.load("FWCore.MessageService.MessageLogger_cfi")

# standard RCT configuration, including input scales
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")

# using standard scales
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")

#include "L1TriggerConfig/L1ScalesProducers/data/L1CaloInputScalesConfig.cff"
process.load("L1Trigger.RegionalCaloTrigger.L1RCTTestAnalyzer_cfi")

process.load("L1Trigger.RegionalCaloTrigger.rctDigis_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(64)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('rct.root')
)

process.source = cms.Source("EmptySource")

process.rctInput = cms.EDProducer("RctInputTextToDigi",
    inputFile = cms.FileInPath('L1Trigger/TextToDigi/test/data/rctTestInputFileElec.txt')
)

process.input = cms.Path(process.rctInput)
process.p4 = cms.Path(process.rctDigis*process.L1RCTTestAnalyzer)
process.schedule = cms.Schedule(process.input,process.p4)

process.L1RCTTestAnalyzer.ecalDigisLabel = 'rctInput'
process.L1RCTTestAnalyzer.hcalDigisLabel = 'rctInput'
process.rctDigis.ecalDigisLabel = 'rctInput'
process.rctDigis.hcalDigisLabel = 'rctInput'


