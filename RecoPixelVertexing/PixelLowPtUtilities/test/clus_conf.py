# Auto generated configuration file
# using:
# Revision: 1.11
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: RE -s DIGI,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO --eventcontent FEVTDEBUG
# --datatier GEN-SIM-DIGI-RAW-RECO --conditions DESIGN61_V10::All --filein file:TenMuE_0_200_cfi_py_GEN_SIM.root -n -1 --geometry ExtendedPhaseIPixel,ExtendedPhaseIPixelReco --customise SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise --no_exec
import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('globalTag', "POSTLS161_V15::All", VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string)
options.parseArguments()

process = cms.Process('ClusterShape')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2017_cff')
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames  = cms.untracked.vstring(options.inputFiles)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.11 $'),
    annotation = cms.untracked.string('RE nevts:-1'),
    name = cms.untracked.string('Applications')
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)

# Additional output definition

# Other statements
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag.globaltag = options.globalTag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')


process.clusterShape = cms.EDAnalyzer("ClusterShapeExtractor",
    trackProducer  = cms.string('allTracks'),
    hasSimHits     = cms.bool(True),
    hasRecTracks   = cms.bool(False),
    associateStrip      = cms.bool(False),
    associatePixel      = cms.bool(True),
    associateRecoTracks = cms.bool(False),
    ROUList = cms.vstring(
      'g4SimHitsTrackerHitsPixelBarrelLowTof',
      'g4SimHitsTrackerHitsPixelBarrelHighTof',
      'g4SimHitsTrackerHitsPixelEndcapLowTof',
      'g4SimHitsTrackerHitsPixelEndcapHighTof')
)

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction*process.clusterShape)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,
                                process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,
                                process.endjob_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.phase1TkCustoms
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise

#call to customisation function customise imported from SLHCUpgradeSimulations.Configuration.phase1TkCustoms
process = customise(process)

# End of customisation functions

process.TFileService = cms.Service('TFileService',
                                  fileName = cms.string("debug.root")
                                   )
