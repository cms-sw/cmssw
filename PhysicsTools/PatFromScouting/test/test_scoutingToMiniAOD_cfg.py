import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# Load particle data table (needed for PF candidate producer)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:store+data+Run2024C+ScoutingPFRun3+HLTSCOUT+v1+000+380+197+00000+05943107-3d39-4fcd-bea9-2b71ca2c3890.root'
    )
)

# Load the scouting to MiniAOD configuration
from PhysicsTools.PatFromScouting.scoutingToMiniAOD_cff import *

process.scoutingPFCandidates = scoutingPFCandidates
process.scoutingVertices = scoutingVertices
process.scoutingMuons = scoutingMuons.clone(
    src = cms.InputTag("hltScoutingMuonPackerVtx")
)
process.scoutingElectrons = scoutingElectrons
process.scoutingPhotons = scoutingPhotons
process.scoutingJets = scoutingJets
process.scoutingMET = scoutingMET
process.scoutingTracks = scoutingTracks

process.p = cms.Path(
    process.scoutingPFCandidates +
    process.scoutingVertices +
    process.scoutingMuons +
    process.scoutingElectrons +
    process.scoutingPhotons +
    process.scoutingJets +
    process.scoutingMET +
    process.scoutingTracks
)

# Output
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('scoutingToMiniAOD_test.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_scoutingPFCandidates_*_*',
        'keep *_scoutingVertices_*_*',
        'keep *_scoutingMuons_*_*',
        'keep *_scoutingElectrons_*_*',
        'keep *_scoutingPhotons_*_*',
        'keep *_scoutingJets_*_*',
        'keep *_scoutingMET_*_*',
        'keep *_scoutingTracks_*_*',
    )
)

process.e = cms.EndPath(process.out)
