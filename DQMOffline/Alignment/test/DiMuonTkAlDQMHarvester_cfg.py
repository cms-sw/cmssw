# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: DiMuonVertex -s HARVESTING:alcaHarvesting --conditions auto:phase1_2022_realistic --mc --filetype DQM --scenario pp --era Run3 -n -1 --filein file:step3_inDQM.root
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('HARVESTING',Run3)

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
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:step3_inDQM_'+options.resonance+'.root')
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    SkipEvent = cms.untracked.vstring(),
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

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('DiMuonVertex nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.dqmsave_step = cms.Path(process.DQMSaver)

if (options.resonance == 'Z'):
    print('',30*"#",'\n # will harvest Z file \n',30*"#")
    _folderName = cms.string('AlCaReco/TkAlDiMuonAndVertex')
    _doBkgFit = cms.bool(True)
    _useRooCBShape = cms.bool(False)
    _useRooCMSShape = cms.bool(True)
    _fitPar =  cms.PSet(mean_par = cms.vdouble(90.,80.,100.),
                        width_par = cms.vdouble(2.49,2.48,2.50),
                        sigma_par = cms.vdouble(1.5,0.,10.))
elif (options.resonance == 'Jpsi'):
    print('',30*"#",'\n # will harvest J/psi file \n',30*"#")
    _folderName =  cms.string('AlCaReco/TkAlJpsiMuMu')
    _doBkgFit = cms.bool(True)
    _useRooCBShape = cms.bool(True)
    _useRooCMSShape = cms.bool(True)
    _fitPar =  cms.PSet(mean_par = cms.vdouble(3.09, 2.7, 3.4),
                        width_par = cms.vdouble(1.0, 0.0, 5.0),
                        sigma_par = cms.vdouble(0.01, 0.0, 5.0))
elif (options.resonance == 'Upsilon'):
    print('',30*"#",'\n # will harvest Upsilon file \n',30*"#")
    _folderName =  cms.string('AlCaReco/TkAlUpsilonMuMu')
    _doBkgFit = cms.bool(True)
    _useRooCBShape = cms.bool(True)
    _useRooCMSShape = cms.bool(False)
    _fitPar =  cms.PSet(mean_par = cms.vdouble(9.46, 8.9, 9.9),
                        width_par = cms.vdouble(1.0, 0.0, 5.0),
                        sigma_par = cms.vdouble(1.0, 0.0, 5.0))

# the module to harvest
process.DiMuonMassBiasClient = cms.EDProducer("DiMuonMassBiasClient",
                                              FolderName = _folderName,
                                              fitBackground = _doBkgFit,
                                              debugMode = cms.bool(True),
                                              fit_par = _fitPar,
                                              useRooCBShape = _useRooCBShape,
                                              useRooCMSShape = _useRooCMSShape,
                                              MEtoHarvest = cms.vstring(
                                                  'DiMuMassVsMuMuPhi',
                                                  'DiMuMassVsMuMuEta',
                                                  'DiMuMassVsMuPlusPhi',
                                                  'DiMuMassVsMuPlusEta',
                                                  'DiMuMassVsMuMinusPhi',
                                                  'DiMuMassVsMuMinusEta',
                                                  'DiMuMassVsMuMuDeltaEta',
                                                  'DiMuMassVsCosThetaCS'
                                              )
                                          )

process.diMuonBiasClient = cms.Sequence(process.DiMuonMassBiasClient)
# trick to run the harvester module
process.alcaHarvesting.insert(1,process.diMuonBiasClient)

# Schedule definition
process.schedule = cms.Schedule(process.alcaHarvesting,process.dqmsave_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
