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
    process.MessageLogger.categories.append('L1TGlobalSummary')
    process.MessageLogger.categories.append('HLTrigReport')

# No longer override - instead use GT config as provided via cmsDriver
## override the GlobalTag, connection string and pfnPrefix
#    if 'GlobalTag' in process.__dict__:
#        process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_CONDITIONS'
#        process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://Frontie#rProd/')
#        
#   process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")

    process=ProcessName(process)

    return(process)


def L1T(process):
#   modifications when running L1T only

    labels = ['gtDigis','simGtDigis','newGtDigis','hltGtDigis']
    for label in labels:
        if label in process.__dict__:
            process.load('L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi')
            process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag( label )
            process.L1AnalyzerEndpath = cms.EndPath( process.l1GtTrigReport )
            process.schedule.append(process.L1AnalyzerEndpath)

    labels = ['gtStage2Digis','simGtStage2Digis','newGtStage2Digis','hltGtStage2Digis']
    for label in labels:
        if label in process.__dict__:
            process.load('L1Trigger.L1TGlobal.L1TGlobalSummary_cfi')
            process.L1TGlobalSummary.AlgInputTag = cms.InputTag( label )
            process.L1TGlobalSummary.ExtInputTag = cms.InputTag( label )
            process.L1TAnalyzerEndpath = cms.EndPath(process.L1TGlobalSummary )
            process.schedule.append(process.L1TAnalyzerEndpath)

    if hasattr(process,'TriggerMenu'):
        delattr(process,'TriggerMenu')

    process=Base(process)

    return(process)


def L1THLT(process):
#   modifications when running L1T+HLT

    if not ('HLTAnalyzerEndpath' in process.__dict__) :
        if 'hltGtDigis' in process.__dict__:
            from HLTrigger.Configuration.HLT_Fake_cff import fragment
            process.hltL1GtTrigReport = fragment.hltL1GtTrigReport
            process.hltTrigReport = fragment.hltTrigReport
            process.HLTAnalyzerEndpath = cms.EndPath(process.hltGtDigis + process.hltL1GtTrigReport + process.hltTrigReport)
            process.schedule.append(process.HLTAnalyzerEndpath)

        if 'hltGtStage2ObjectMap' in process.__dict__:
            from HLTrigger.Configuration.HLT_FULL_cff import fragment
            process.hltL1TGlobalSummary = fragment.hltL1TGlobalSummary
            process.hltTrigReport = fragment.hltTrigReport
            process.HLTAnalyzerEndpath = cms.EndPath(process.hltGtStage2Digis + process.hltL1TGlobalSummary + process.hltTrigReport)
            process.schedule.append(process.HLTAnalyzerEndpath)

    if hasattr(process,'TriggerMenu'):
        delattr(process,'TriggerMenu')

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

def L1REPACK(process,sequence="Full"):

    from Configuration.StandardSequences.Eras import eras

    l1repack = cms.Process('L1REPACK',eras.Run2_2016)
    l1repack.load('Configuration.StandardSequences.SimL1EmulatorRepack_'+sequence+'_cff')

    for module in l1repack.es_sources_():
        if (not hasattr(process,module)):
            setattr(process,module,getattr(l1repack,module))
    for module in l1repack.es_producers_():
        if (not hasattr(process,module)):
            setattr(process,module,getattr(l1repack,module))

    for module in l1repack.SimL1Emulator.expandAndClone().moduleNames():
        setattr(process,module,getattr(l1repack,module))
    process.SimL1Emulator = l1repack.SimL1Emulator

    for path in process.paths_():
        getattr(process,path).insert(0,process.SimL1Emulator)
    for path in process.endpaths_():
        getattr(process,path).insert(0,process.SimL1Emulator)

    return process

def L1XML(process,xmlFile=None):

#   xmlFile="L1Menu_Collisions2016_dev_v3.xml"

    if ((xmlFile is None) or (xmlFile=="")):
        return process

    process.L1TriggerMenu= cms.ESProducer("L1TUtmTriggerMenuESProducer",
        L1TriggerMenuFile= cms.string(xmlFile)
    )
    process.ESPreferL1TXML = cms.ESPrefer("L1TUtmTriggerMenuESProducer","L1TriggerMenu")

    return process
