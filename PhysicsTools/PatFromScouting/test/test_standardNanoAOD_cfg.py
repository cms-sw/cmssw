"""
Test standard NanoAOD production on scouting MiniAOD.

This is the recommended production workflow:
1. First run test_scoutingToMiniAOD_cfg.py to create scoutingToMiniAOD_test.root
2. Then run this config to produce standard NanoAOD

This uses cmsDriver-style configuration with our customizations.
"""

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_nanoAOD_pre142X_cff import run3_nanoAOD_pre142X

process = cms.Process('NANO', Run3, run3_nanoAOD_pre142X)

# ============================================================
# Standard configurations
# ============================================================

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('PhysicsTools.NanoAOD.nano_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 10

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# ============================================================
# Input: Scouting MiniAOD
# ============================================================

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/work/CMSSW_15_0_18/src/scoutingToMiniAOD_test.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    TryToContinue = cms.untracked.vstring('ProductNotFound'),
)

# ============================================================
# Global Tag
# ============================================================

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data_prompt', '')

# ============================================================
# Output
# ============================================================

process.NANOAODoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAOD'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:standardNanoAOD_test.root'),
    outputCommands = process.NANOAODEventContent.outputCommands
)

# ============================================================
# Paths
# ============================================================

process.nanoAOD_step = cms.Path(process.nanoSequence)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.NANOAODoutput_step = cms.EndPath(process.NANOAODoutput)

process.schedule = cms.Schedule(
    process.nanoAOD_step,
    process.endjob_step,
    process.NANOAODoutput_step
)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# ============================================================
# Apply customizations
# ============================================================

# Standard NanoAOD customization
from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeCommon
process = nanoAOD_customizeCommon(process)

# Scouting-specific customizations
from PhysicsTools.PatFromScouting.nanoAOD_scouting_cff import customiseNanoForScoutingMiniAOD
process = customiseNanoForScoutingMiniAOD(process)

# ============================================================
# Additional fixes for scouting MiniAOD
# ============================================================

# Add additional rho producers that NanoAOD expects
# (our MiniAOD has fixedGridRhoFastjetAll, but NanoAOD also needs fixedGridRhoFastjetCentral)
process.fixedGridRhoFastjetCentral = cms.EDProducer("FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag("packedPFCandidates"),
    maxRapidity = cms.double(2.5),  # Central only
    gridSpacing = cms.double(0.55)
)

# Add to the beginning of nanoAOD path
process.nanoAOD_step.insert(0, process.fixedGridRhoFastjetCentral)

print("=" * 60)
print("Standard NanoAOD on Scouting MiniAOD")
print("=" * 60)
print("Input: scoutingToMiniAOD_test.root")
print("Output: standardNanoAOD_test.root")
print("=" * 60)
