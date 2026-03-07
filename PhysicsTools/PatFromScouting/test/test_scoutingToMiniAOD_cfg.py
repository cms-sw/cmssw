import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.inputFiles = ['file:/data/dmytro/data/store+data+Run2024C+ScoutingPFRun3+HLTSCOUT+v1+000+380+197+00000+05943107-3d39-4fcd-bea9-2b71ca2c3890.root']
options.maxEvents = 100
options.parseArguments()

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# Load particle data table (needed for PF candidate producer)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

# Load conditions for beam spot
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data_prompt', '')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles)
)

# Load the scouting to MiniAOD configuration
from PhysicsTools.PatFromScouting.scoutingToMiniAOD_cff import *

# Use standard MiniAOD collection names
process.packedPFCandidates = packedPFCandidates
process.offlineSlimmedPrimaryVertices = offlineSlimmedPrimaryVertices
process.slimmedMuons = slimmedMuons.clone(
    src = cms.InputTag("hltScoutingMuonPackerVtx")
)
process.slimmedMuonsNoVtx = slimmedMuonsNoVtx
process.scoutingDimuonVertices = scoutingDimuonVertices.clone(
    scoutingMuons = cms.InputTag("hltScoutingMuonPackerVtx"),
    scoutingVertices = cms.InputTag("hltScoutingMuonPackerVtx", "displacedVtx"),
)
process.scoutingDimuonVerticesNoVtx = scoutingDimuonVerticesNoVtx
process.slimmedElectrons = slimmedElectrons
process.slimmedPhotons = slimmedPhotons
process.slimmedJets = slimmedJets
process.slimmedMETs = slimmedMETs
process.scoutingTracks = scoutingTracks
process.offlineBeamSpot = offlineBeamSpot
process.fixedGridRhoFastjetAll = fixedGridRhoFastjetAll
process.gtStage2Digis = gtStage2Digis
process.gmtStage2Digis = gmtStage2Digis
process.caloStage2Digis = caloStage2Digis

process.p = cms.Path(
    process.offlineSlimmedPrimaryVertices +
    process.scoutingTracks +
    process.packedPFCandidates +
    process.offlineBeamSpot +
    process.slimmedMuons +
    process.slimmedMuonsNoVtx +
    process.scoutingDimuonVertices +
    process.scoutingDimuonVerticesNoVtx +
    process.slimmedElectrons +
    process.slimmedPhotons +
    process.slimmedJets +
    process.slimmedMETs +
    process.fixedGridRhoFastjetAll +
    process.gtStage2Digis +
    process.gmtStage2Digis +
    process.caloStage2Digis
)

# Output
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('scoutingToMiniAOD_test.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_packedPFCandidates_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',
        'keep *_slimmedMuons_*_*',
        'keep *_slimmedMuonsNoVtx_*_*',
        'keep *_scoutingDimuonVertices_*_*',
        'keep *_scoutingDimuonVerticesNoVtx_*_*',
        'keep *_slimmedElectrons_*_*',
        'keep *_slimmedPhotons_*_*',
        'keep *_slimmedJets_*_*',
        'keep *_slimmedMETs_*_*',
        'keep *_scoutingTracks_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_fixedGridRho*_*_*',
        'keep *_gtStage2Digis_*_*',
        'keep *_gmtStage2Digis_*_*',
        'keep *_caloStage2Digis_*_*',
    )
)

process.e = cms.EndPath(process.out)
