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

# last re-reco /MinimumBias/BeamCommissioning09-SD_AllMinBias-Dec19thSkim_341_v1/RAW-RECO
# dbs search --query="find run where dataset=/MinimumBias/BeamCommissioning09-SD_AllMinBias-Dec19thSkim_341_v1/RAW-RECO"
# dbs search --query="find file where dataset=/MinimumBias/BeamCommissioning09-SD_AllMinBias-Dec19thSkim_341_v1/RAW-RECO and run=124020"

### For 219, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/E63696DD-A5ED-DE11-9217-00261894382D.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/E03CA059-ACED-DE11-9ECF-00261894388D.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/DAC9A273-AEED-DE11-9AFE-00261894397F.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/DAB63AE3-A5ED-DE11-B770-0026189438F2.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/BCF4F55B-ACED-DE11-A210-0030486791DC.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/AA6AF32B-A8ED-DE11-9FD5-003048D3C010.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/9279CC40-AAED-DE11-87A4-00261894392C.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/9080ADD7-A5ED-DE11-A8D2-00261894388D.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/82020FFB-A7ED-DE11-88BD-003048678FF8.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/7CA3E946-AAED-DE11-A018-00304867906C.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/76EEB0DF-A5ED-DE11-9BED-001A92971AD0.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/62A14048-AAED-DE11-A08A-002618943884.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/603F01D7-A5ED-DE11-B9B0-00304867BED8.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/5469908D-B0ED-DE11-8355-002618943867.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/4C701344-AAED-DE11-BF8D-0026189438A7.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/46663C28-A8ED-DE11-A1D9-002618943945.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/36777C8E-B0ED-DE11-A8AC-002618FDA28E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/3258955B-ACED-DE11-B201-002618943809.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/306184F6-A7ED-DE11-AE6C-003048678F06.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/0CC8B05A-ACED-DE11-88F5-00261894382D.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/0A7FD35A-ACED-DE11-A58F-00261894397F.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0005/04C8B343-AAED-DE11-B79E-00248C0BE014.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/E2F293FB-90ED-DE11-9E96-001A92811732.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/D25B7D86-9BED-DE11-B5D7-003048678C06.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/BE98BC7D-9BED-DE11-95E4-001731EF61B4.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/A67DE9FD-9EED-DE11-9996-003048678A88.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/98C25867-99ED-DE11-A143-002618FDA28E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/96E47027-97ED-DE11-BF05-003048678B38.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/8C9D405C-9DED-DE11-8A33-00304867906C.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/8AB23F5E-9DED-DE11-8E26-0018F3D09688.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/7083B20A-9FED-DE11-914C-00261894383A.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/6C3C8E01-95ED-DE11-8279-003048678FE0.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/605782F3-90ED-DE11-B5E7-002618943950.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/386B095B-9DED-DE11-B20A-002618FDA28E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/2AEB4383-9BED-DE11-8720-003048679220.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/1C5F7A27-97ED-DE11-BE35-001A92971ACC.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec19thSkim_341_v1/0004/16E12B7D-9BED-DE11-99CF-003048678B8E.root')
)
# add after dataset name in crab job
# runselection = XXXXXX, YYYYYYY, .....

# Analyzer module
process.myanalysis = cms.EDFilter(
    "JPTAnalyzer_Data",
    HistOutFile      = cms.untracked.string('analysis.root'),
#    calojets         = cms.string('iterativeCone5CaloJets'),
#    calojets         = cms.string('sisCone5CaloJets'),
    calojets         = cms.string('ak5CaloJets'),
    jetsID           = cms.string('ak5JetID'),
    jetExtender      = cms.string('ak5JetExtender'),
#    zspjets          = cms.string('ZSPJetCorJetIcone5'),
#    zspjets          = cms.string('ZSPJetCorJetSiscone5'),
    zspjets          = cms.string('ZSPJetCorJetAntiKt5'),
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorIcone5')
#    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorSiscone5')
    JetCorrectionJPT = cms.string('JetPlusTrackZSPCorrectorAntiKt5')
    )

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

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

