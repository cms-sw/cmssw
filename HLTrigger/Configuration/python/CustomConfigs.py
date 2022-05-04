import FWCore.ParameterSet.Config as cms

# The following 2 imports are provided for backward compatibility reasons.
# The functions used to be defined in this file.
from FWCore.ParameterSet.MassReplace import massReplaceInputTag as MassReplaceInputTag
from FWCore.ParameterSet.MassReplace import massReplaceParameter as MassReplaceParameter

def ProcessName(process):
#   processname modifications

    if 'hltTrigReport' in process.__dict__:
        process.hltTrigReport.HLTriggerResults = cms.InputTag( 'TriggerResults','',process.name_() )

    return(process)


def Base(process):
#   default modifications

    process.options.wantSummary = cms.untracked.bool(True)
    process.options.numberOfThreads = cms.untracked.uint32( 4 )
    process.options.numberOfStreams = cms.untracked.uint32( 0 )
    process.options.sizeOfStackForThreadsInKB = cms.untracked.uint32( 10*1024 )

    process.MessageLogger.TriggerSummaryProducerAOD=cms.untracked.PSet()
    process.MessageLogger.L1GtTrigReport=cms.untracked.PSet()
    process.MessageLogger.L1TGlobalSummary=cms.untracked.PSet()
    process.MessageLogger.HLTrigReport=cms.untracked.PSet()

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

    def _legacyStage1(process):
        labels = ['gtDigis','simGtDigis','newGtDigis','hltGtDigis']
        for label in labels:
            if label in process.__dict__:
                process.load('L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi')
                process.l1GtTrigReport.L1GtRecordInputTag = cms.InputTag( label )
                process.L1AnalyzerEndpath = cms.EndPath( process.l1GtTrigReport )
                process.schedule.append(process.L1AnalyzerEndpath)

    def _stage2(process):
        labels = ['gtStage2Digis','simGtStage2Digis','newGtStage2Digis','hltGtStage2Digis']
        for label in labels:
            if label in process.__dict__:
                process.load('L1Trigger.L1TGlobal.L1TGlobalSummary_cfi')
                process.L1TGlobalSummary.AlgInputTag = cms.InputTag( label )
                process.L1TGlobalSummary.ExtInputTag = cms.InputTag( label )
                process.L1TAnalyzerEndpath = cms.EndPath(process.L1TGlobalSummary )
                process.schedule.append(process.L1TAnalyzerEndpath)

    from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
    (~stage2L1Trigger).toModify(process, _legacyStage1)
    stage2L1Trigger.toModify(process, _stage2)

    if hasattr(process,'TriggerMenu'):
        delattr(process,'TriggerMenu')

    process=Base(process)

    return(process)


def L1THLT(process):
#   modifications when running L1T+HLT

    if not ('HLTAnalyzerEndpath' in process.__dict__) :
        def _legacyStage1(process):
            if 'hltGtDigis' in process.__dict__:
                from HLTrigger.Configuration.HLT_Fake_cff import fragment
                process.hltL1GtTrigReport = fragment.hltL1GtTrigReport
                process.hltTrigReport = fragment.hltTrigReport
                process.HLTAnalyzerEndpath = cms.EndPath(process.hltGtDigis + process.hltL1GtTrigReport + process.hltTrigReport)
                process.schedule.append(process.HLTAnalyzerEndpath)

        def _stage2(process):
            if 'hltGtStage2ObjectMap' in process.__dict__:
                from HLTrigger.Configuration.HLT_FULL_cff import fragment
                process.hltL1TGlobalSummary = fragment.hltL1TGlobalSummary
                process.hltTrigReport = fragment.hltTrigReport
                process.HLTAnalyzerEndpath = cms.EndPath(process.hltGtStage2Digis + process.hltL1TGlobalSummary + process.hltTrigReport)
                process.schedule.append(process.HLTAnalyzerEndpath)

        from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
        (~stage2L1Trigger).toModify(process, _legacyStage1)
        stage2L1Trigger.toModify(process, _stage2)

    if hasattr(process,'TriggerMenu'):
        delattr(process,'TriggerMenu')

    process=Base(process)

    return(process)


def HLTRECO(process):

    # --------------------------------------------
    # remove ESSources and ESProducers from Tasks:
    #  - needed for HLT+RECO tests on GPU
    # --------------------------------------------
    #  - when Reconstruction_cff is loaded, it brings in Tasks that include
    #    GPU-related ES modules with the same names as they have in HLT configs
    #  - in TSG tests, these GPU-related RECO Tasks are not included in the Schedule
    #    (because the "gpu" process-modifier is not used);
    #    this causes the ES modules not to be executed, thus making them unavailable to HLT producers
    #  - this workaround removes ES modules from Tasks, making their execution independent of the content of the Schedule;
    #    with reference to https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideAboutPythonConfigFile?rev=92#Behavior_when_an_ESProducer_ESSo,
    #    this workaround avoids "Case 3" by reverting to "Case 2"
    #  - this workaround only affects Tasks of non-HLT steps, as the addition of ES modules to Tasks is not supported in ConfDB
    #    (none of the Tasks used in the HLT step can contain ES modules in the first place, modulo customisations outside ConfDB)
    for taskName in process.tasks_():
        task = process.tasks_()[taskName]
        esModulesToRemove = set()
        for modName in task.moduleNames():
            module = getattr(process, modName)
            if isinstance(module, cms.ESSource) or isinstance(module, cms.ESProducer):
                esModulesToRemove.add(module)
        for esModule in esModulesToRemove:
            task.remove(esModule)
    # --------------------------------------------

    return process


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


def L1REPACK(process, sequence="Full"):

    from Configuration.Eras.Era_Run3_cff import Run3
    l1repack = cms.Process('L1REPACK', Run3)
    l1repack.load('Configuration.StandardSequences.SimL1EmulatorRepack_'+sequence+'_cff')

    for module in l1repack.es_sources_():
        if not hasattr(process, module):
            setattr(process, module, getattr(l1repack, module))
    for module in l1repack.es_producers_():
        if not hasattr(process, module):
            setattr(process, module, getattr(l1repack, module))

    for module in l1repack.SimL1Emulator.expandAndClone().moduleNames():
        setattr(process, module, getattr(l1repack, module))
    for taskName, task in l1repack.tasks_().items():
        if l1repack.SimL1Emulator.contains(task):
            setattr(process, taskName, task)
    for sequenceName, sequence in l1repack.sequences_().items():
        if l1repack.SimL1Emulator.contains(sequence):
            setattr(process, sequenceName, sequence)

    process.SimL1Emulator = l1repack.SimL1Emulator

    for path in process.paths_():
        getattr(process,path).insert(0,process.SimL1Emulator)
    for path in process.endpaths_():
        getattr(process,path).insert(0,process.SimL1Emulator)

    # special L1T cleanup
    for obj in [
      'hgcalTriggerGeometryESProducer',
    ]:
        if hasattr(process, obj):
            delattr(process, obj)

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
