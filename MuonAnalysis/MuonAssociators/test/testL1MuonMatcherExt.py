import FWCore.ParameterSet.Config as cms

process = cms.Process("TestL1MatcherExtended")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_R_36X_V12::All'


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/jpsi/CS_Onia-Jun14thSkim_v1_RAW-RECO_run136082_443584C2-B27E-DF11-9E13-0017A477001C.root'
    )
)

process.arbMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isTrackerMuon && track.numberOfValidHits >= 12 && numberOfMatches > 0"),
)

process.load("MuonAnalysis.MuonAssociators.muonL1MatchExtended_cfi")
process.muonL1MatchExtended.muons = "arbMuons"

import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi
process.patMuons = PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.clone(
    muonSource = 'arbMuons',
    embedTrack          = True,
    embedCombinedMuon   = True,
    embedStandAloneMuon = True,
    embedPickyMuon = False,
    embedTpfmsMuon = False, 
    userIsolation = cms.PSet(), # no extra isolation
    isoDeposits = cms.PSet(),   # no isodeposits
    addGenMatch = False,        # no mc
)

from MuonAnalysis.MuonAssociators.muonL1MatchExtended_cfi import addUserData as addMuonL1MatchExtended
addMuonL1MatchExtended(process.patMuons, addExtraInfo=True)

process.s = cms.Sequence(
    process.arbMuons +
    process.csctfDigis +
    process.muonL1MatchExtended +
    process.patMuons
)

process.p = cms.Path(
    process.s
)

process.o = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("patMuons_L1MatcherExtended.root"),
    outputCommands = cms.untracked.vstring("drop *", "keep *_patMuons__*", "keep l1extraL1MuonParticles_l1extraParticles__*")
)
process.e = cms.EndPath(process.o)

