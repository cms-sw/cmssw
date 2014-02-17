# Auto generated configuration file
# using: 
# Revision: 1.100 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt10.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,RECO -n 3 --eventcontent FEVTDEBUGHLT --conditions FrontierConditions_GlobalTag,IDEAL_30X::All --relval 10000,100 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/Digi_cff')
process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.load('Configuration/StandardSequences/L1TriggerDefaultMenu_cff')
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

# gavril : add CalibTracker/SiPixelErrorEstimation module
process.load("CalibTracker/SiPixelErrorEstimation/SiPixelErrorEstimation_cfi")
# gavril
process.PixelErrorEstimation.outputFile = cms.untracked.string('./SiPixelErrorEstimation_ntuple.root')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('SingleMuPt10.cfi nevts:3'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MinPhi = cms.untracked.double(-3.14159265359),
        MinPt = cms.untracked.double(9.99),
        PartID = cms.untracked.vint32(-13, 13, -13, 13, -13, -13, 13, -13, 13, -13),
        MaxEta = cms.untracked.double(2.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.5),
        MaxPt = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single mu pt 10'),
    AddAntiParticle = cms.untracked.bool(True),
    firstRun = cms.untracked.uint32(1)
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_RECO.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'IDEAL_30X::All'

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

process.PixelErrorEstimation = cms.Path(process.PixelErrorEstimation) # gavril : add PixelErrorEstimation module

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.digitisation_step,
                                process.L1simulation_step,
                                process.digi2raw_step,
                                process.raw2digi_step,
                                process.reconstruction_step,
                                process.PixelErrorEstimation, # gavril: add PixelErrorEstimation module in the path
                                process.endjob_step,
                                process.out_step
                                )
