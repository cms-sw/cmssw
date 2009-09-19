import FWCore.ParameterSet.Config as cms

process = cms.Process("readelectrons")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring
    (
        '/store/mc/Summer09/Zee/GEN-SIM-RECO/MC_31X_V3_preproduction_312-v1/0008/B4E5948E-1679-DE11-8721-002219205095.root',
        '/store/mc/Summer09/Zee/GEN-SIM-RECO/MC_31X_V3_preproduction_312-v1/0006/B28F900A-0B79-DE11-B2D3-00D0680BF9A4.root',
    ),
    secondaryFileNames = cms.untracked.vstring(),
)

process.mergedSuperClusters = cms.EDFilter("EgammaSuperClusterMerger",
    src = cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"), cms.InputTag("multi5x5SuperClustersWithPreshower"))
)

from RecoEgamma.Examples.dataAnalyzerStdBiningParameters_cff import *
from RecoEgamma.Examples.dataAnalyzerFineBiningParameters_cff import *

process.gsfElectronAnalysis = cms.EDAnalyzer("GsfElectronDataAnalyzer",
    electronCollection = cms.InputTag("gsfElectrons"),
    matchingObjectCollection = cms.InputTag("mergedSuperClusters"),
    readAOD = cms.bool(False),
    outputFile = cms.string('gsfElectronHistos_data_Summer09Zee_new.root'),
    MaxPt = cms.double(100.0),
    DeltaR = cms.double(0.3),
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
    MaxTkIso03 = cms.double(1.0),
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


