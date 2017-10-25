import os
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras


def get_root_files(path):
    files = os.listdir(path)
    root_files = [f for f in files if f.endswith(".root")]
    full_paths = [os.path.join(path, f) for f in root_files]
    urls = ['file://{0}'.format(f) for f in full_paths]
    return urls


options = VarParsing('analysis')
options.register(
    'sample',
    'TTJet',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
)

options.setDefault('maxEvents', 2000)
options.setDefault(
    'outputFile', 'L1TOffline_L1TStage2CaloLayer2_job1_RAW2DIGI_RECO_DQM.root')

options.parseArguments()

inputFiles = {
    'TTJet': get_root_files('/data/TTJet/reco'),
    'DoubleEG': get_root_files('/data/DoubleEG'),
}

inputFilesRAW = {
    'TTJet': get_root_files('/data/TTJet/raw'),
}


process = cms.Process('L1TStage2EmulatorDQM', eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load(
    'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = int(options.maxEvents / 10)

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(inputFiles[options.sample]),
)
if options.sample == 'TTJet':
    process.source.secondaryFileNames = cms.untracked.vstring(inputFilesRAW[
                                                              'TTJet'])

process.options = cms.untracked.PSet(

)

# Output definition
process.DQMoutput = cms.OutputModule(
    "DQMRootOutputModule",
    fileName=cms.untracked.string(options.outputFile)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
if options.sample == 'TTJet':
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
else:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)

process.load('DQMOffline.L1Trigger.L1TStage2CaloLayer2Offline_cfi')
process.load('DQMOffline.L1Trigger.L1TEGammaOffline_cfi')
process.load('DQMOffline.L1Trigger.L1TTauOffline_cfi')

if os.environ.get('DEBUG', False):
    process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
    process.MessageLogger.debugModules = cms.untracked.vstring(
        '*',
    )

process.dqmoffline_step = cms.Path(
    process.l1tStage2CaloLayer2OfflineDQMEmu +
    process.l1tStage2CaloLayer2OfflineDQM +
    process.l1tEGammaOfflineDQM +
    process.l1tEGammaOfflineDQMEmu +
    process.l1tTauOfflineDQM +
    process.l1tTauOfflineDQMEmu
)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
# Schedule definition
process.schedule = cms.Schedule(
    process.raw2digi_step,
)

# customisation of the process.

# Automatic addition of the customisation function from
# L1Trigger.Configuration.customiseReEmul
from L1Trigger.Configuration.customiseReEmul import L1TReEmulFromRAW

# call to customisation function L1TReEmulFromRAW imported from
# L1Trigger.Configuration.customiseReEmul
# complains about
# AttributeError: 'Process' object has no attribute 'simRctDigis'
# process = L1TReEmulFromRAW(process)
process.schedule.append(process.dqmoffline_step)
process.schedule.append(process.DQMoutput_step)
