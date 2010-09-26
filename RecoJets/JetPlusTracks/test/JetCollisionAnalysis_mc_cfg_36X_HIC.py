import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO4")

process.load('Configuration/StandardSequences/Services_cff')
process.load("Configuration/StandardSequences.ReconstructionHeavyIons_cff")
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.Simulation_cff")
#process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/L1Reco_cff')
#process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('DQMOffline/Configuration/DQMOffline_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.load('Configuration/StandardSequences.MixingNoPileUp_cff')
process.load('Configuration/StandardSequences.VtxSmearedGauss_cff')

process.load('RecoJets/Configuration/RecoJPTJetsHIC_cff')

#===>set qualityBit to hiSelectedTracks
process.hiSelectedTracks.qualityBit = cms.string('loose')
#===>

process.GlobalTag.globaltag = cms.string('MC_38Y_V7::All')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
### For CMSSW_3_8_0, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring( 
##'/store/relval/CMSSW_3_6_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/0410D2F0-AD44-DF11-9F84-001A9281172E.root'

#'/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/0056B15C-265D-DF11-AE6D-0018F3D09660.root'

'/store/relval/CMSSW_3_8_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0004/0CD2F4D0-1C95-DF11-B8F0-003048D42D92.root'
)
)

#===> ak5, ak5
#process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
#    HistOutFile = cms.untracked.string('JetAnalysis.root'),
#    src2 = cms.InputTag("ak5GenJets"),
#    src22 = cms.InputTag("ak5GenJets"),
#    src3 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
##    src4 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt7"),
#    src4 = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),    
#    src1 = cms.InputTag("ak5CaloJets"),
#    src11 = cms.InputTag("ak5CaloJets"),
#    Data = cms.int32(0),
#    jetsID = cms.string('ak5JetID'),
#    jetsID2 = cms.string('ak5JetID'),
#    Cone1 = cms.double(0.5),
#    Cone2 = cms.double(0.5),
#    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
#    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
#    HORecHitCollectionLabel = cms.InputTag("horeco"),
#    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
#    inputTrackLabel = cms.untracked.string('generalTracks')
#)

#===> ic5
process.myjetplustrack = cms.EDAnalyzer("JetPlusTrackAnalysis",
##process.myjetplustrack = cms.EDFilter("JetPlusTrackAnalysis",
    HistOutFile = cms.untracked.string('JetAnalysis.root'),
    src2 = cms.InputTag("iterativeCone5GenJets"),
#    src22  = cms.InputTag("ak5GenJets"),
    src3  = cms.InputTag("JetPlusTrackZSPCorJetIconePu5"),
#    src4  = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
    src1 = cms.InputTag("iterativeConePu5CaloJets"),
#    src11  = cms.InputTag("ak5CaloJets"),     
    Data  = cms.int32(0),
    jetsID = cms.string('ic5JetID'),
#    jetsID2 = cms.string('ak5JetID'),
    Cone1 = cms.double(0.5),
#    Cone2 = cms.double(0.5),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HFRecHitCollectionLabel = cms.InputTag("hfreco"),
    HORecHitCollectionLabel = cms.InputTag("horeco"),
    HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
    inputTrackLabel = cms.untracked.string('hiSelectedTracks')
)

process.rawtodigi_step = cms.Path(process.RawToDigi)
process.reco = cms.Path(process.reconstructionHeavyIons)
process.jptbg = cms.Path(process.recoJPTJetsHIC)
process.myan = cms.Path(process.myjetplustrack)

process.schedule = cms.Schedule(process.rawtodigi_step,process.reco,process.jptbg,process.myan)

## orig ## process.p1 = cms.Path(process.recoJPTJets)

#process.p1 = cms.Path(process.reconstructionHeavyIons*process.recoJPTJetsHIC*process.myjetplustrack)

#process.p1 = cms.Path(process.runjets*process.recoJPTJetsHIC*process.myjetplustrack)
#
#### re-reco of jet-track association
#process.p1 = cms.Path(process.ak5JTA*process.ak7JTA*process.JetPlusTrackCorrectionsAntiKt5*process.JetPlusTrackCorrectionsAntiKt7*process.myjetplustrack)

