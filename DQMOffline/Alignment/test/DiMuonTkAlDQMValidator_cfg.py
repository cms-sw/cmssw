import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('reRECO',Run3)

import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('maxEvents',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "maximum events")
options.register('globalTag',
                 'auto:phase1_2022_realistic',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "conditions")
options.register('resonance',
                 'Z',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "resonance type")
options.parseArguments()

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 200

process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

process.load("DQMOffline.Configuration.AlCaRecoDQM_cff")

if (options.resonance == 'Z'):
    # Z
    print('',30*"#",'\n # will process Z->mm data\n',30*"#")
    readFiles = ['/store/relval/CMSSW_12_5_0_pre2/RelValZMM_14/ALCARECO/TkAlDiMuonAndVertex-124X_mcRun3_2022_realistic_v3-v1/2580000/4f9aee02-35a2-49b7-93f5-831214cf32d8.root']
    process.seqALCARECOTkAlDQM = cms.Sequence(process.ALCARECOTkAlDiMuonAndVertexVtxDQM + process.ALCARECOTkAlDiMuonMassBiasDQM)
elif (options.resonance == 'Jpsi'):
    # J/psi
    print('',30*"#",'\n # will process Jpsi->mm data\n',30*"#")
    readFiles = ['/store/relval/CMSSW_12_5_0_pre2/RelValEtaBToJpsiJpsi_14TeV/ALCARECO/TkAlJpsiMuMu-124X_mcRun3_2022_realistic_v3-v1/2580000/fc77aaed-b0f5-4446-87f5-f7341099bd73.root']
    process.seqALCARECOTkAlDQM = cms.Sequence(process.ALCARECOTkAlJpsiMuMuVtxDQM + process.ALCARECOTkAlJpsiMassBiasDQM)
elif (options.resonance == 'Upsilon'):
    # upsilon
    print('',30*"#",'\n # will process Upsilon->mm data\n',30*"#")
    readFiles = ['/store/relval/CMSSW_12_5_0_pre2/RelValUpsilon1SToMuMu_14/ALCARECO/TkAlUpsilonMuMu-124X_mcRun3_2022_realistic_v3-v1/2580000/fca73916-5076-4c9f-a460-2481588825ab.root']
    process.seqALCARECOTkAlDQM = cms.Sequence(process.ALCARECOTkAlUpsilonMuMuVtxDQM + process.ALCARECOTkAlUpsilonMassBiasDQM)
else:
    print('unrecongnized %s resonance',options.resonance)
    exit(1)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(readFiles),
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM_'+options.resonance+'.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.dqmoffline_step = cms.EndPath(process.seqALCARECOTkAlDQM)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.schedule = cms.Schedule(process.dqmoffline_step,process.DQMoutput_step)
