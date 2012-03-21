import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration import customizeHLTforL1Emulator


def ProcessName(process):
#   processname modifications

    if 'hltTrigReport' in process.__dict__:
        process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    if 'hltDQMHLTScalers' in process.__dict__:
        process.hltDQMHLTScalers.triggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    if 'hltDQML1SeedLogicScalers' in process.__dict__:
        process.hltDQML1SeedLogicScalers.processname = process.name_()

    return(process)


def Base(process):
#   default modifications

    process.options.wantSummary = cms.untracked.bool(True)

    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')

    process=ProcessName(process)

    return(process)


def L1T(process):
#   modifications when running L1T only

    process.load('L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi')
    process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag( "simGtDigis" )

    process.L1AnalyzerEndpath = cms.EndPath( process.l1GtTrigReport )
    process.schedule.append(process.L1AnalyzerEndpath)

    process=Base(process)

    return(process)


def L1THLT(process):
#   modifications when running L1T+HLT

    if not ('HLTAnalyzerEndpath' in process.__dict__) :
        from HLTrigger.Configuration.HLT_FULL_cff import hltL1GtTrigReport,hltTrigReport
        process.hltL1GtTrigReport = hltL1GtTrigReport
        process.hltTrigReport = hltTrigReport
        process.HLTAnalyzerEndpath = cms.EndPath(process.hltL1GtTrigReport +  process.hltTrigReport)
        process.schedule.append(process.HLTAnalyzerEndpath)

    process=Base(process)

    return(process)


#def L1THLT2(process):
##   modifications when re-running L1T+HLT    
#
##   run trigger primitive generation on unpacked digis, then central L1
#
#    process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")
#    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
#    process.simHcalTriggerPrimitiveDigis.inputLabel = ('hcalDigis', 'hcalDigis')
#
##   patch the process to use 'sim*Digis' from the L1 emulator
##   instead of 'hlt*Digis' from the RAW data
#
#    patchToRerunL1Emulator.switchToSimGtDigis( process )
#
#    process=Base(process)
#
#    return(process)


def HLTDropPrevious(process):
#   drop on input the previous HLT results
    process.source.inputCommands = cms.untracked.vstring (
        'keep *',
        'drop *_hltL1GtObjectMap_*_*',
        'drop *_TriggerResults_*_*',
        'drop *_hltTriggerSummaryAOD_*_*',
    )

    process=Base(process)
    
    return(process)
