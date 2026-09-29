"""
Example configuration for testing offline HGCAL NanoAOD production.

Usage:
  cmsRun testHGCalNanoAOD_cfg.py inputFiles=file:step3_RECO.root maxEvents=100

This config shows how to run the offline HGCAL NanoAOD table producers
starting from RECO files.
"""

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# Setup command line options
options = VarParsing('analysis')
options.register('skipEvents', 0, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Number of events to skip")
options.parseArguments()

# Create the process
from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('HGCALNANO', Phase2C17I13M9)

# Import standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('PhysicsTools.NanoAOD.nano_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Message logger
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles),
    skipEvents=cms.untracked.uint32(options.skipEvents),
)

# Global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Load HGCAL NanoAOD configuration
from DPGAnalysis.HGCalNanoAOD.HGCalNanoAOD_cff import hgcalNanoValidationSequence, hgcalNanoCustomize

# Add the HGCAL NanoAOD sequence
process.hgcalNanoAOD = cms.Path(hgcalNanoValidationSequence)

# Output definition
process.NANOAODSIMoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm=cms.untracked.string('LZMA'),
    compressionLevel=cms.untracked.int32(9),
    dataset=cms.untracked.PSet(
        dataTier=cms.untracked.string('NANOAODSIM'),
        filterName=cms.untracked.string('')
    ),
    fileName=cms.untracked.string('file:hgcal_nano.root'),
    outputCommands=cms.untracked.vstring(
        'drop *',
        'keep nanoaodFlatTable_*Table*_*_*',
    )
)

# Output path
process.endjob_step = cms.EndPath(process.endOfProcess)
process.NANOAODSIMoutput_step = cms.EndPath(process.NANOAODSIMoutput)

# Schedule
process.schedule = cms.Schedule(
    process.hgcalNanoAOD,
    process.endjob_step,
    process.NANOAODSIMoutput_step
)

# Apply customization
process = hgcalNanoCustomize(process)

# Dump configuration (optional)
if options.maxEvents < 0 or options.maxEvents > 1000:
    process.options = cms.untracked.PSet(
        wantSummary=cms.untracked.bool(True)
    )
