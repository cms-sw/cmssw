import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO4")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/L1Reco_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('DQMOffline/Configuration/DQMOffline_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")

#### Choose techical bits 40 or 41 and coincidence with BPTX (0)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND (NOT 36 AND NOT 37 AND NOT 38 AND NOT 39)')
####
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
#          '/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/596/F494AB9A-40E2-DE11-8D1E-000423D33970.root'
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/F08F782B-77E8-DE11-B1FC-0019B9F72BFF.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/EE9412FD-80E8-DE11-9FDD-000423D94908.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/7C9741F5-78E8-DE11-8E69-001D09F2AD84.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/44255E49-80E8-DE11-B6DB-000423D991F0.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/3C02A810-7CE8-DE11-BB51-003048D375AA.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04F15557-7BE8-DE11-8A41-003048D2C1C4.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04092AB7-75E8-DE11-958F-000423D98750.root'
         'file:./reco900GeV123596.root'
)
)

#process.myjetplustrack = cms.EDAnalyzer("JetPlusTrackAnalysis",
#    HistOutFile = cms.untracked.string('JetAnalysis.root'),
#    src2 = cms.InputTag("iterativeCone5GenJets"),
#    src3 = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
#    src4 = cms.InputTag("ZSPJetCorJetIcone5"),
#    src1 = cms.InputTag("iterativeCone5CaloJets"),
#    Cone = cms.double(0.5),
#    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
#    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
#    HORecHitCollectionLabel = cms.InputTag("horeco"),
#    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
#    inputTrackLabel = cms.untracked.string('generalTracks')
#)

#### remove monster events
process.monster = cms.EDFilter(
    "FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(True),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.2)
    )
####



process.myjetplustrack = cms.EDAnalyzer("JetPlusTrackCollisionAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src1 = cms.InputTag("ak5CaloJets"),
    src2 = cms.InputTag("ZSPJetCorJetAntiKt5"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    Cone = cms.double(0.5),
    JPTname = cms.untracked.string('JetPlusTrackZSPCorrectorAntiKt5'),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
#    HORecHitCollectionLabel = cms.InputTag("horeco"),
#    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HBHENZSRecHitCollectionLabel = cms.InputTag("hbherecoMB"),
    inputTrackLabel = cms.untracked.string('generalTracks')
)


process.p1 = cms.Path(process.monster*process.ZSPJetCorrectionsAntiKt5*process.JetPlusTrackCorrectionsAntiKt5*process.myjetplustrack)
#process.p1 = cms.Path(process.hltLevel1GTSeed*process.ZSPJetCorrectionsAntiKt5*process.JetPlusTrackCorrectionsAntiKt5*process.myjetplustrack)
