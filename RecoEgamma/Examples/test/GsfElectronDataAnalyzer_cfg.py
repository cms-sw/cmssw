import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring
    (
        '/store/mc/Summer09/Zee/GEN-SIM-RECO/MC_31X_V3-v1/0028/DAA8926E-CE8B-DE11-9654-0030487DF78A.root',
        '/store/mc/Summer09/Zee/GEN-SIM-RECO/MC_31X_V3-v1/0027/E6394B5A-338B-DE11-A710-00151764221C.root',
        '/store/mc/Summer09/Zee/GEN-SIM-RECO/MC_31X_V3-v1/0027/B00D1565-438B-DE11-A443-001E682F8528.root'
    ),
    secondaryFileNames = cms.untracked.vstring(),
)

process.mergedSuperClusters = cms.EDProducer("EgammaSuperClusterMerger",
    src = cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"), cms.InputTag("multi5x5SuperClustersWithPreshower"))
)

from RecoEgamma.Examples.dataAnalyzerStdBiningParameters_cff import *
from RecoEgamma.Examples.dataAnalyzerFineBiningParameters_cff import *

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronDataAnalyzer",
    beamSpot = cms.InputTag('offlineBeamSpot'),
    electronCollection = cms.InputTag("gsfElectrons"),
    readAOD = cms.bool(False),
    outputFile = cms.string('gsfElectronHistos_data_ZeeSummer09.root'),
    triggerResults = cms.InputTag("TriggerResults::HLT"),
    hltPaths = cms.vstring('HLT_Ele10_SW_L1R','HLT_Ele15_SW_L1R','HLT_Ele15_SW_EleId_L1R','HLT_Ele15_SW_LooseTrackIso_L1R','HLT_Ele15_SC15_SW_LooseTrackIso_L1R','HLT_Ele15_SC15_SW_EleId_L1R','HLT_Ele20_SW_L1R','HLT_Ele20_SC15_SW_L1R','HLT_Ele25_SW_L1R','HLT_Ele25_SW_EleId_LooseTrackIso_L1R','HLT_DoubleEle10_SW_L1R'),
    matchingObjectCollection = cms.InputTag("mergedSuperClusters"),
    matchingCondition = cms.string("Cone"),
    DeltaR = cms.double(0.3),
    MaxPtMatchingObject = cms.double(100.0),
    MaxAbsEtaMatchingObject = cms.double(2.5),
    MinEt = cms.double(4.),
    MinPt = cms.double(0.),
    MaxAbsEta = cms.double(2.5),
    SelectEB = cms.bool(False),
    SelectEE = cms.bool(False),
    SelectNotEBEEGap = cms.bool(False),
    SelectEcalDriven = cms.bool(False),
    SelectTrackerDriven = cms.bool(False),
    MinEOverPBarrel = cms.double(0.),
    MaxEOverPBarrel = cms.double(10000.),
    MinEOverPEndcaps = cms.double(0.),
    MaxEOverPEndcaps = cms.double(10000.),
    MinDetaBarrel = cms.double(0.),
    MaxDetaBarrel = cms.double(10000.),
    MinDetaEndcaps = cms.double(0.),
    MaxDetaEndcaps = cms.double(10000.),
    MinDphiBarrel = cms.double(0.),
    MaxDphiBarrel = cms.double(10000.),
    MinDphiEndcaps = cms.double(0.),
    MaxDphiEndcaps = cms.double(10000.),
    MinSigIetaIetaBarrel = cms.double(0.),
    MaxSigIetaIetaBarrel = cms.double(10000.),
    MinSigIetaIetaEndcaps = cms.double(0.),
    MaxSigIetaIetaEndcaps = cms.double(10000.),
    MaxHoEBarrel = cms.double(10000.),
    MaxHoEEndcaps = cms.double(10000.),
    MinMVA = cms.double(-10000.),
    MaxTipBarrel = cms.double(10000.),
    MaxTipEndcaps = cms.double(10000.),
    MaxTkIso03 = cms.double(10000.),
    MaxHcalIso03Depth1Barrel = cms.double(10000.),
    MaxHcalIso03Depth1Endcaps = cms.double(10000.),
    MaxHcalIso03Depth2Endcaps = cms.double(10000.),
    MaxEcalIso03Barrel = cms.double(10000.),
    MaxEcalIso03Endcaps = cms.double(10000.),
    HistosConfigurationData = cms.PSet(
    dataAnalyzerStdBiningParameters
    #dataAnalyzerFineBiningParameters
    )
)

process.p = cms.Path(process.mergedSuperClusters*process.gsfElectronAnalysis)


