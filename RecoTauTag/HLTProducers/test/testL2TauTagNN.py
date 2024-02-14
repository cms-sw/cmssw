import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3
import os
import re

options = VarParsing('analysis')
options.register('sampleType', 'Run3MC', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Indicates the sample type: Run3MC or Run2Data")
options.register('fileList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of root files to process.")
options.register('fileNamePrefix', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "Prefix to add to input file names.")
options.register('lumiFile', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "JSON file with lumi mask.")
options.register('eventList', '', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "List of events to process.")
options.register('dumpPython', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "Dump full config into stdout.")
options.register('Summary', False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "want summary")

options.parseArguments()
dataDict = {"Run3MC" : False, "Run2Data" : True}
isData = dataDict[options.sampleType]
processName = 'MLProva'
process = cms.Process(processName, Run3)
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

if not isData:
    process.load('SimGeneral.MixingModule.mixNoPU_cfi')

process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())

# Input source
def readFileList(fileList, inputFileName, fileNamePrefix):
    with open(inputFileName, 'r') as inputFile:
        for name in inputFile.readlines():
            if len(name) > 0 and name[0] != '#':
                fileList.append(fileNamePrefix + name)

def addFilesToList(fileList, inputFiles, fileNamePrefix):
    """read intput file list from a another list"""
    for name in inputFiles:
        if len(name) > 0 and name[0] != '#':
            fileList.append(fileNamePrefix + name)

if len(options.fileList) > 0:
    readFileList(process.source.fileNames, options.fileList, options.fileNamePrefix)
elif len(options.inputFiles) > 0:
    addFilesToList(process.source.fileNames, options.inputFiles, options.fileNamePrefix)
else:
    #print(options.inputFiles)
    #print(options.fileList)

    #process.source.fileNames = cms.untracked.vstring('file:/store/mc/Run3Winter20DRPremixMiniAOD/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8/GEN-SIM-RAW/110X_mcRun3_2021_realistic_v6-v1/20000/1E71AB01-1FF5-F049-94D4-642325AFF937.root')  # 5000 evts

    #process.source.fileNames = cms.untracked.vstring('/store/mc/Run3Winter20DRPremixMiniAOD/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8/GEN-SIM-RAW/110X_mcRun3_2021_realistic_v6-v1/20000/A95D23CB-88AC-6849-8980-5D12C8417007.root') # 11000 evts
    #process.source.fileNames = cms.untracked.vstring('file:/eos/home-v/vdamante/A95D23CB-88AC-6849-8980-5D12C8417007.root') # 11000 evts
    process.source.fileNames = cms.untracked.vstring('/store/mc/Run3Winter21DRMiniAOD/VBFHToTauTau_M125_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/FlatPU30to80FEVT_112X_mcRun3_2021_realistic_v16-v1/270000/005b56c1-0107-46b3-9740-1c6efc559295.root') #1000 evts
    #process.source.fileNames = cms.untracked.vstring('file:/store/mc/Run3Winter20DRPremixMiniAOD/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8/GEN-SIM-RAW/110X_mcRun3_2021_realistic_v6-v1/20000/F2760E46-A3DB-DA4F-A6EC-525C10EDCBC7.root') # 1000 evts
    #process.source.fileNames = cms.untracked.vstring('file:/store/mc/Run3Winter20DRPremixMiniAOD/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8/GEN-SIM-RAW/110X_mcRun3_2021_realistic_v6-v1/20000/657F58E7-64E3-AA4D-B505-6D0F39997487.root')



if len(options.lumiFile) > 0:
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = options.lumiFile).getVLuminosityBlockRange()

if options.eventList != '':
    process.source.eventsToProcess = cms.untracked.VEventRange(re.split(',', options.eventList))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)




process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    #TryToContinue = cms.untracked.vstring('ProductNotFound'),
    TryToContinue = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(1)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)


# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('tau_hlt nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
GlobalTagName = 'auto:run3_data_GRun' if isData else 'auto:run3_mc_GRun'
process.GlobalTag = GlobalTag(process.GlobalTag, GlobalTagName, '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)


# Schedule definition
# process.schedule imported from cff in HLTrigger.Configuration
process.schedule.extend([process.endjob_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)


# customisation of the process.


# Automatic addition of the customisation function

if isData:
    # Customisation from command line
    from HLTrigger.Configuration.customizeHLTforCMSSW import customiseFor2018Input
    process = customiseFor2018Input(process)
else:
    from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
    process = customizeHLTforMC(process)


#from applyL2TauTag import update
from RecoTauTag.HLTProducers.applyL2TauTag import update
process = update(process)
#process.schedule = cms.Schedule(*[ process.HLTriggerFirstPath, process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4, process.HLTriggerFinalPath, process.endjob_step ], tasks=[process.patAlgosToolsTask])


# End of customisation functions

# Customisation from command line

process.load('FWCore.MessageLogger.MessageLogger_cfi')
x = process.maxEvents.input.value()
x = x if x >= 0 else 10000
process.MessageLogger.cerr.FwkReport.reportEvery = max(1, min(1000, x // 10))

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

if options.dumpPython:
    print (process.dumpPython())

process.options.wantSummary = cms.untracked.bool(options.Summary)


# End adding early deletion
