import FWCore.ParameterSet.Config as cms

process = cms.Process("PATMuon")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticlesNoSimHits_cfi")    # On RECO
process.load("SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi")  # On RECO

process.GlobalTag.globaltag = 'START3X_V26A::All'

from Configuration.EventContent.EventContent_cff import *
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/MC/Feb9Skims/MC_CollisionEvents_MuonSkim_1.root',
        #'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/MC/Feb9Skims/MC_CollisionEvents_MuonSkim_2.root',
        #'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/MC/Feb9Skims/MC_CollisionEvents_MuonSkim_3.root',
        'root://pcmssd12.cern.ch//data/gpetrucc/Feb9Skims/MC_CollisionEvents_MuonSkim_1.root'
    ),
    inputCommands = RECOSIMEventContent.outputCommands,            # keep only RECO out of RAW+RECO, for tests
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),  # keep only RECO out of RAW+RECO, for tests
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.bscFilter = hltLevel1GTSeed.clone(L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)'))
process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof >= 4 && abs(z) <= 15 && position.Rho <= 2"),
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)
process.noScraping = cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
)
process.countCollisionEvents = cms.EDProducer("EventCountProducer")
process.preFilter = cms.Sequence(process.bscFilter * process.oneGoodVertexFilter * process.noScraping * process.countCollisionEvents )

process.mergedTruth = cms.EDProducer("GenPlusSimParticleProducer",
        src           = cms.InputTag("g4SimHits"), # use "famosSimHits" for FAMOS
        setStatus     = cms.int32(5),             # set status = 8 for GEANT GPs
        filter        = cms.vstring("pt > 0.0"),  # just for testing (optional)
        genParticles   = cms.InputTag("genParticles") # original genParticle list
)
process.genMuons = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
        "drop  *  ",                     # this is the default
        "++keep abs(pdgId) = 13",        # keep muons and their parents
        "drop pdgId == 21 && status = 2" # remove intermediate qcd spam carrying no flavour info
    )
)
process.load("PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi")

process.filter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("muons"),
    minNumber = cms.uint32(1),
)

import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi
process.patMuons = PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.clone(
    muonSource = 'muons',
    # embed the tracks, so we don't have to carry them around
    embedTrack          = True,
    embedCombinedMuon   = True,
    embedStandAloneMuon = True,
    # then switch off some features we don't need
    #addTeVRefits = False, ## <<--- this doesn't work. PAT bug ??
    embedPickyMuon = False,
    embedTpfmsMuon = False, 
    userIsolation = cms.PSet(),   # no extra isolation beyond what's in reco::Muon itself
    isoDeposits = cms.PSet(), # no heavy isodeposits
    addGenMatch = False,       
    embedGenMatch = False,
)

process.load("MuonAnalysis.MuonAssociators.muonClassificationByHits_cfi")

from MuonAnalysis.MuonAssociators.muonClassificationByHits_cfi import addUserData as addClassByHits
addClassByHits(process.patMuons, extraInfo=True)

# as an example, now we define yet another classification, only for TMLastStationLoose.
# (the selection matters when you define ghosts)
process.classByHitsTMLSLoose = process.classByHitsTM.clone(
    muonPreselection = cms.string("muonID('TMLastStationLoose')")
)
addClassByHits(process.patMuons, labels=["classByHitsTMLSLoose"], extraInfo=True)


process.go = cms.Path(
    process.preFilter +
    process.filter    +
    ( process.mergedTruth *
      process.genMuons    *
      process.muonMatch   +
      process.muonClassificationByHits +
      process.classByHitsTMLSLoose ) *
    process.patMuons  
)


process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("patMuons_MC.root"),
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep *_patMuons_*_*",
        "keep *_genMuons_*_*",
        "keep *_countCollisionEvents_*_*",
        "keep recoTrackExtras_standAloneMuons_*_*",          ## track states at the muon system, used both by patMuons and standAloneMuons
        "keep recoTracks_standAloneMuons__*",                ## bare standalone muon tracks, using standalone muon momentum (without BS constraint)
        "keep edmTriggerResults_*_*_HLT",                    ## 
        "keep l1extraL1MuonParticles_l1extraParticles_*_*",  ## 
        "keep *_offlinePrimaryVertices__*",                  ## 
        "keep *_offlineBeamSpot__*",                         ##
    ),
    SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring("go")),
)
process.end = cms.EndPath(process.out)
