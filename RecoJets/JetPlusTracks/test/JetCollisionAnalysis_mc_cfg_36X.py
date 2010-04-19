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


process.load('RecoJets.Configuration.RecoJPTJets_cff')


process.GlobalTag.globaltag = cms.string('MC_36Y_V4::All')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_36Y_V4-v1/0011/1A6E87A4-AD44-DF11-BDAA-0018F3D09684.root'
)
)

process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src2 = cms.InputTag("ak5GenJets"),
    src22 = cms.InputTag("ak7GenJets"),
    src3 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    src4 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
#    src4 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),    
    src1 = cms.InputTag("ak5CaloJets"),
    src11 = cms.InputTag("ak7CaloJets"),
    Data = cms.int32(0),
    jetsID = cms.string('ak5JetID'),
    jetsID2 = cms.string('ak5JetID'),
    Cone1 = cms.double(0.5),
    Cone2 = cms.double(0.7),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    inputTrackLabel = cms.untracked.string('generalTracks')
)

process.p1 = cms.Path(process.recoJPTJets)
#process.p1 = cms.Path(process.recoJPTJets*process.myjetplustrack)
#
#### re-reco of jet-track association
#process.p1 = cms.Path(process.ak5JTA*process.ak7JTA*process.JetPlusTrackCorrectionsAntiKt5*process.JetPlusTrackCorrectionsAntiKt7*process.myjetplustrack)

