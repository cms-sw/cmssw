import FWCore.ParameterSet.Config as cms

muonTriggerRateTimeAnalyzer = cms.EDAnalyzer("MuonTriggerRateTimeAnalyzer",

    HltProcessName = cms.string("HLT"),
    
    UseMuonFromGenerator = cms.bool(True),
    UseMuonFromReco      = cms.bool(True),
    GenLabel  = cms.untracked.string('genParticles'),
    RecoLabel = cms.untracked.string('globalMuons'),

    NtuplePath         = cms.untracked.string('hltSingleMuIso'),
    NtupleFileName     = cms.untracked.string(''),
    RootFileName       = cms.untracked.string(''),

    MaxPtParameters    = cms.vdouble(40,0.,40.),
    PtParameters       = cms.vdouble(50,0.,1000.),
    EtaParameters      = cms.vdouble(50,-2.1,2.1),
    PhiParameters      = cms.vdouble(50,-3.15,3.15),

    MinPtCut           = cms.untracked.double(10.0),
    MaxEtaCut          = cms.untracked.double(2.1),
    L1DrCut            = cms.untracked.double(0.4),
    L2DrCut            = cms.untracked.double(0.25),
    L3DrCut            = cms.untracked.double(0.015),
    MotherParticleId   = cms.untracked.uint32(0),
    TriggerResultLabel = cms.InputTag("TriggerResults","","HLT"),

    DQMStore = cms.untracked.bool(True),
    CrossSection = cms.double(0.97),
    NSigmas90 = cms.untracked.vdouble(3.0, 3.0, 3.0, 3.0),

    TimerLabel = cms.InputTag("hltTimer"),
    TimeNbins = cms.uint32(150),
    MaxTime = cms.double(150.0),
    TriggerCollection = cms.VPSet(
        cms.PSet(
            L1ReferenceThreshold = cms.double(7.0),
            HltCollectionLabels = cms.vstring(
                "hltSingleMuIsoL2PreFiltered",
                "hltSingleMuIsoL2IsoFiltered",
                "hltSingleMuIsoL3PreFiltered",
                "hltSingleMuIsoL3IsoFiltered"),
            L1CollectionLabel = cms.string("hltSingleMuIsoL1Filtered"),
            HltReferenceThreshold = cms.double(7.0),
            NumberOfObjects = cms.uint32(1)
    ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(7.0),
            HltCollectionLabels = cms.vstring(
                "hltSingleMuNoIsoL2PreFiltered",
                "hltSingleMuNoIsoL3PreFiltered"),
            L1CollectionLabel = cms.string("hltSingleMuNoIsoL1Filtered"),
            HltReferenceThreshold = cms.double(7.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HltCollectionLabels = cms.vstring(
                "hltDiMuonNoIsoL2PreFiltered",
                "hltDiMuonNoIsoL3PreFiltered"),
            L1CollectionLabel = cms.string("hltDiMuonNoIsoL1Filtered"),
            HltReferenceThreshold = cms.double(7.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HltCollectionLabels = cms.vstring(
                "hltDiMuonIsoL2PreFiltered",
                "hltDiMuonIsoL2IsoFiltered",
                "hltDiMuonIsoL3PreFiltered",
                "hltDiMuonIsoL3IsoFiltered"),
            L1CollectionLabel = cms.string("hltDiMuonIsoL1Filtered"),
            HltReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HltCollectionLabels = cms.vstring(
                "hltJpsiMML2Filtered",
                "hltJpsiMML3Filtered"),
            L1CollectionLabel = cms.string("hltJpsiMML1Filtered"),
            HltReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HltCollectionLabels = cms.vstring(
                "hltZMML2Filtered",
                "hltZMML3Filtered"),
            L1CollectionLabel = cms.string("hltZMML1Filtered"),
            HltReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HltCollectionLabels = cms.vstring(
                "hltSingleMuPrescale3L2PreFiltered",
                "hltSingleMuPrescale3L3PreFiltered"),
            L1CollectionLabel = cms.string("hltSingleMuPrescale3L1Filtered"),
            HltReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HltCollectionLabels = cms.vstring(
                "hltSingleMuPrescale5L2PreFiltered",
                "hltSingleMuPrescale5L3PreFiltered"),
            L1CollectionLabel = cms.string("hltSingleMuPrescale5L1Filtered"),
            HltReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(7.0),
            HltCollectionLabels = cms.vstring(
                "hltSingleMuPrescale77L2PreFiltered",
                "hltSingleMuPrescale77L3PreFiltered"),
            L1CollectionLabel = cms.string("hltSingleMuPrescale77L1Filtered"),
            HltReferenceThreshold = cms.double(5.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(7.0),
            HltCollectionLabels = cms.vstring(
                "hltSingleMuPrescale710L2PreFiltered",
                "hltSingleMuPrescale710L3PreFiltered"),
            L1CollectionLabel = cms.string("hltSingleMuPrescale1710L1Filtered"),
            HltReferenceThreshold = cms.double(7.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HltCollectionLabels = cms.vstring(),
            L1CollectionLabel = cms.string("hltMuLevel1PathL1Filtered"),
            HltReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(1)
        )
    ),
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
