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

#
# No longer override - instead use GT config as provided via cmsDriver
## override the GlobalTag, connection string and pfnPrefix
#    if 'GlobalTag' in process.__dict__:
#        process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_CONDITIONS'
#        process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://Frontie#rProd/')
#        
    process=ProcessName(process)

    return(process)


def L1T(process):
#   modifications when running L1T only

    process.load('L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi')
    labels = ['gtDigis','simGtDigis','newGtDigis','hltGtDigis']
    for label in labels:
        if label in process.__dict__:
            process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag( label )

    process.L1AnalyzerEndpath = cms.EndPath( process.l1GtTrigReport )
    process.schedule.append(process.L1AnalyzerEndpath)

    process=Base(process)

    return(process)


def L1THLT(process):
#   modifications when running L1T+HLT

    if not ('HLTAnalyzerEndpath' in process.__dict__) :
        from HLTrigger.Configuration.HLT_FULL_cff import fragment
        process.hltL1GtTrigReport = fragment.hltL1GtTrigReport
        process.hltTrigReport = fragment.hltTrigReport
        process.HLTAnalyzerEndpath = cms.EndPath(process.hltL1GtTrigReport + process.hltTrigReport)
        process.schedule.append(process.HLTAnalyzerEndpath)

    process=Base(process)

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


def MassReplaceInputTag(process,old="rawDataCollector",new="rawDataRepacker",verbose=False,moduleLabelOnly=False,skipLabelTest=False):
#   replace InputTag values (adapted from Configuration/Applications/python/ConfigBuilder.py)
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag
    for s in process.paths_().keys():
        massSearchReplaceAnyInputTag(getattr(process,s),old,new,verbose,moduleLabelOnly,skipLabelTest)
    for s in process.endpaths_().keys():
        massSearchReplaceAnyInputTag(getattr(process,s),old,new,verbose,moduleLabelOnly,skipLabelTest)
    return(process)

def MassReplaceParameter(process,name="label",old="rawDataCollector",new="rawDataRepacker",verbose=False):
#   replace values of named parameters
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceParam
    for s in process.paths_().keys():
        massSearchReplaceParam(getattr(process,s),name,old,new,verbose)
    for s in process.endpaths_().keys():
        massSearchReplaceParam(getattr(process,s),name,old,new,verbose)
    return(process)

def L1REPACK(process):
#   Replace only the L1 parts and keep the rest
    if 'DigiToRaw' in process.__dict__:
        process.DigiToRaw = cms.Sequence(process.l1tDigiToRawSeq + process.l1GtPack + process.l1GtEvmPack + process.rawDataCollector)
    if 'rawDataCollector' in process.__dict__:
        process.rawDataCollector.RawCollectionList = cms.VInputTag(
            cms.InputTag('gctDigiToRaw'),
            cms.InputTag('l1tDigiToRaw'),
            cms.InputTag('l1GtPack'),
            cms.InputTag('l1GtEvmPack'),
            cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
        )

    process=L1T(process)

    return process
