# ak5JetExtender
# ak5JetID

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

#### Choose techical bits 40 and coincidence with BPTX (0)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND 40 AND NOT (36 OR 37 OR 38 OR 39)')
#### remove monster events
process.monster = cms.EDFilter(
    "FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(True),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.2)
    )
####
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
# last re-reco /MinimumBias/BeamCommissioning09-SD_AllMinBias-Dec19thSkim_341_v1
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
#         '/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/596/F494AB9A-40E2-DE11-8D1E-000423D33970.root'
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/E63696DD-A5ED-DE11-9217-00261894382D.root'
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/F08F782B-77E8-DE11-B1FC-0019B9F72BFF.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/EE9412FD-80E8-DE11-9FDD-000423D94908.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/7C9741F5-78E8-DE11-8E69-001D09F2AD84.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/44255E49-80E8-DE11-B6DB-000423D991F0.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/3C02A810-7CE8-DE11-BB51-003048D375AA.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04F15557-7BE8-DE11-8A41-003048D2C1C4.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/124/120/04092AB7-75E8-DE11-958F-000423D98750.root'
#         'file:./RECOHcalCalMinBias.root'
)
)

# Analyzer module
process.myanalysis = cms.EDFilter(
    "JPTAnalyzer_Data",
    HistOutFile      = cms.untracked.string('analysis.root'),
#    calojets         = cms.string('iterativeCone5CaloJets'),
#    calojets         = cms.string('sisCone5CaloJets'),
    calojets         = cms.string('ak5CaloJets'),
#    zspjets          = cms.string('ZSPJetCorJetIcone5'),
#    zspjets          = cms.string('ZSPJetCorJetSiscone5'),
    zspjets          = cms.string('ZSPJetCorJetAntiKt5'),
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorIcone5')
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorSiscone5')
    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorAntiKt5')
    )

process.dump = cms.EDFilter("EventContentAnalyzer")

# Path
process.p1 = cms.Path(
#   process.dump *
   process.hltLevel1GTSeed *
   process.monster *
#    process.ZSPJetCorrectionsIcone5 *
#    process.ZSPrecoJetAssociationsIcone5 *
#    process.ZSPJetCorrectionsSisCone5 *
#    process.ZSPrecoJetAssociationsSisCone5 *
    process.ZSPJetCorrectionsAntiKt5 *	
    process.ZSPrecoJetAssociationsAntiKt5 *
    process.myanalysis
    )

