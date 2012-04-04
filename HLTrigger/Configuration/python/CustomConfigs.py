import FWCore.ParameterSet.Config as cms

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

# override the GlobalTag, connection string and pfnPrefix
    if 'GlobalTag' in process.__dict__:
        process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
        process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
        
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


def FASTSIM(process):
#   modifications when running L1T+HLT

    process=L1THLT(process)
    process.hltL1GtTrigReport.L1GtRecordInputTag = cms.InputTag("gtDigis")

    return(process)


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
