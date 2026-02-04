"""
Scouting NanoAOD production test.

This produces NanoAOD from scouting MiniAOD using the custom scouting tables.

Prerequisites:
    First run test_scoutingToMiniAOD_cfg.py to create scoutingToMiniAOD_test.root

Usage:
    cmsRun test_scoutingNanoAOD_cfg.py

Output:
    scoutingNanoAOD_test.root - NanoAOD format with physics objects and triggers
"""

import FWCore.ParameterSet.Config as cms

process = cms.Process("NANO")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# Load conditions for L1 trigger menu
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data_prompt', '')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:scoutingToMiniAOD_test.root'
    )
)

# Load the scouting NanoAOD configuration
from PhysicsTools.PatFromScouting.scoutingNanoAOD_cff import (
    scoutingMuonTable,
    scoutingElectronTable,
    scoutingPhotonTable,
    scoutingJetTable,
    scoutingMETTable,
    scoutingPVTable,
    scoutingEventTable,
    l1bits,
)

process.scoutingMuonTable = scoutingMuonTable.clone()
process.scoutingElectronTable = scoutingElectronTable.clone()
process.scoutingPhotonTable = scoutingPhotonTable.clone()
process.scoutingJetTable = scoutingJetTable.clone()
process.scoutingMETTable = scoutingMETTable.clone()
process.scoutingPVTable = scoutingPVTable.clone()
process.scoutingEventTable = scoutingEventTable.clone()
process.l1bits = l1bits.clone()

process.p = cms.Path(
    process.scoutingMuonTable +
    process.scoutingElectronTable +
    process.scoutingPhotonTable +
    process.scoutingJetTable +
    process.scoutingMETTable +
    process.scoutingPVTable +
    process.scoutingEventTable +
    process.l1bits
)

# NanoAOD output
process.out = cms.OutputModule("NanoAODOutputModule",
    fileName = cms.untracked.string('scoutingNanoAOD_test.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep nanoaodFlatTable_*_*_*',
        'keep edmTriggerResults_*_*_*',
    ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
)

process.e = cms.EndPath(process.out)
