import FWCore.ParameterSet.Config as cms

muonTriggerRateTimeAnalyzer = cms.EDAnalyzer("MuonTriggerRateTimeAnalyzer",

    HltProcessName = cms.string("HLT"),

    # To disable gen or reco matching, set to an empty string
    GenLabel   = cms.untracked.string('genParticles'), 
    RecoLabel  = cms.untracked.string('globalMuons'),

    # If the RAW trigger summary is unavailable, you can use the AOD instead
    UseAod     = cms.untracked.bool(False),
    AodL1Label = cms.untracked.string('hltMuLevel1PathL1Filtered'),
    AodL2Label = cms.untracked.string('hltSingleMuLevel2NoIsoL2PreFiltered'),

    # Save variables to an ntuple for more in-depth trigger studies
    NtuplePath         = cms.untracked.string('HLT_IsoMu9'),
    NtupleFileName     = cms.untracked.string(''),
    RootFileName       = cms.untracked.string(''),

    # Set the ranges and numbers of bins for histograms
    MaxPtParameters =  cms.vdouble(0,
                          1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                         45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                         110, 120, 140, 170, 200, 250, 300, 380, 500
                       ), 
    PtParameters       = cms.vdouble(50,0.,1000.),
    EtaParameters      = cms.vdouble(42,-2.1,2.1),
    PhiParameters      = cms.vdouble(50,-3.142,3.142),

    # Set cuts placed on the generated muons and matching criteria
    # Use pt cut just below 10 to allow through SingleMuPt10 muons  
    MinPtCut           = cms.untracked.double(9.9),
    MaxEtaCut          = cms.untracked.double(2.1),
    MotherParticleId   = cms.untracked.uint32(0),
    L1DrCut            = cms.untracked.double(0.4),
    L2DrCut            = cms.untracked.double(0.25),
    L3DrCut            = cms.untracked.double(0.015),
    TriggerResultLabel = cms.InputTag("TriggerResults","","HLT"),

    DQMStore = cms.untracked.bool(True),
    CrossSection = cms.double(0.97),
    NSigmas90 = cms.untracked.vdouble(3.0, 3.0, 3.0, 3.0),

    # This timing module is deprecated
    TimerLabel = cms.InputTag("hltTimer"),
    TimeNbins = cms.uint32(150),
    MaxTime = cms.double(150.0),
    TimingModules = cms.untracked.PSet(
        MuonL3IsoModules = cms.untracked.vstring('pixelTracks', 
            'hltL3MuonIsolations', 
            'SingleMuIsoL3IsoFiltered'),
        TrackerRecModules = cms.untracked.vstring('siPixelClusters', 
            'siPixelRecHits', 
            'siStripClusters', 
            'siStripMatchedRecHits'),
        MuonLocalRecModules = cms.untracked.vstring('dt1DRecHits', 
            'dt4DSegments', 
            'rpcRecHits', 
            'csc2DRecHits', 
            'cscSegments'),
        CaloDigiModules = cms.untracked.vstring('ecalDigis', 
            'ecalPreshowerDigis', 
            'hcalDigis'),
        MuonL3RecModules = cms.untracked.vstring('hltL3Muons', 
            'hltL3MuonCandidates', 
            'SingleMuIsoL3PreFiltered'),
        CaloRecModules = cms.untracked.vstring('ecalRegionalMuonsWeightUncalibRecHit', 
            'ecalRegionalMuonsRecHit', 
            'ecalPreshowerRecHit', 
            'hbhereco', 
            'horeco', 
            'hfreco', 
            'towerMakerForMuons', 
            'caloTowersForMuons'),
        MuonL2IsoModules = cms.untracked.vstring('hltL2MuonIsolations', 
            'SingleMuIsoL2IsoFiltered'),
        MuonDigiModules = cms.untracked.vstring('muonCSCDigis', 
            'muonDTDigis', 
            'muonRPCDigis'),
        TrackerDigiModules = cms.untracked.vstring('siPixelDigis', 
            'siStripDigis'),
        MuonL2RecModules = cms.untracked.vstring('hltL2MuonSeeds', 
            'hltL2Muons', 
            'hltL2MuonCandidates', 
            'SingleMuIsoL2PreFiltered')
    )
)
