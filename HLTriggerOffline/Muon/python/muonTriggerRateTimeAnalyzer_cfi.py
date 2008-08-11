import FWCore.ParameterSet.Config as cms

muonTriggerRateTimeAnalyzer = cms.EDAnalyzer("MuonTriggerRateTimeAnalyzer",
    RecoLabel = cms.untracked.InputTag("globalMuons"),
    TriggerCollection = cms.VPSet(cms.PSet(
        L1ReferenceThreshold = cms.double(7.0),
        HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltSingleMuIsoL2PreFiltered","","HLT"), cms.InputTag("hltSingleMuIsoL2IsoFiltered","","HLT"), cms.InputTag("hltSingleMuIsoL3PreFiltered","","HLT"), cms.InputTag("hltSingleMuIsoL3IsoFiltered","","HLT")),
        L1CollectionLabel = cms.InputTag("hltSingleMuIsoL1Filtered","","HLT"),
        HLTReferenceThreshold = cms.double(7.0),
        NumberOfObjects = cms.uint32(1)
    ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(7.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltSingleMuNoIsoL2PreFiltered","","HLT"), cms.InputTag("hltSingleMuNoIsoL3PreFiltered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltSingleMuNoIsoL1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(7.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltDiMuonNoIsoL2PreFiltered","","HLT"), cms.InputTag("hltDiMuonNoIsoL3PreFiltered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltDiMuonNoIsoL1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(7.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltDiMuonIsoL2PreFiltered","","HLT"), cms.InputTag("hltDiMuonIsoL2IsoFiltered","","HLT"), cms.InputTag("hltDiMuonIsoL3PreFiltered","","HLT"), cms.InputTag("hltDiMuonIsoL3IsoFiltered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltDiMuonIsoL1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltJpsiMML2Filtered","","HLT"), cms.InputTag("hltJpsiMML3Filtered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltJpsiMML1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltZMML2Filtered","","HLT"), cms.InputTag("hltZMML3Filtered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltZMML1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(2)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltSingleMuPrescale3L2PreFiltered","","HLT"), cms.InputTag("hltSingleMuPrescale3L3PreFiltered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltSingleMuPrescale3L1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltSingleMuPrescale5L2PreFiltered","","HLT"), cms.InputTag("hltSingleMuPrescale5L3PreFiltered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltSingleMuPrescale5L1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(7.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltSingleMuPrescale77L2PreFiltered","","HLT"), cms.InputTag("hltSingleMuPrescale77L3PreFiltered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltSingleMuPrescale77L1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(5.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(7.0),
            HLTCollectionLabels = cms.VInputTag(cms.InputTag("hltSingleMuPrescale710L2PreFiltered","","HLT"), cms.InputTag("hltSingleMuPrescale710L3PreFiltered","","HLT")),
            L1CollectionLabel = cms.InputTag("hltSingleMuPrescale1710L1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(7.0),
            NumberOfObjects = cms.uint32(1)
        ), 
        cms.PSet(
            L1ReferenceThreshold = cms.double(3.0),
            HLTCollectionLabels = cms.VInputTag(),
            L1CollectionLabel = cms.InputTag("hltMuLevel1PathL1Filtered","","HLT"),
            HLTReferenceThreshold = cms.double(3.0),
            NumberOfObjects = cms.uint32(1)
        )),
    UseMuonFromGenerator = cms.bool(True),
    UseMuonFromReco = cms.bool(True),
    Nbins = cms.untracked.uint32(40),
    GenLabel = cms.untracked.InputTag("source"),
    Luminosity = cms.untracked.double(1e+32),
    DQMStore = cms.untracked.bool(True),
    FolderName = cms.untracked.string('HLT/Muon/'),
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
    ),
    RootFileName = cms.untracked.string(''),
    NSigmas90 = cms.untracked.vdouble(3.0, 3.0, 3.0, 3.0),
    PtMin = cms.untracked.double(0.0),
    CrossSection = cms.double(0.97),
    TimerLabel = cms.InputTag("hltTimer"),
    MaxTime = cms.double(150.0),
    PtMax = cms.untracked.double(40.0),
    TriggerResultLabel = cms.InputTag("TriggerResults","","HLT"),
    TimeNbins = cms.uint32(150)
)



