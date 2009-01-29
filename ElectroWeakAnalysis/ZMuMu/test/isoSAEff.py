import FWCore.ParameterSet.Config as cms

process = cms.Process("IsoSAEff")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_1.root"
    "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_2.root"
    "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_3.root"
    "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_4.root"
    )
)

process.isoAnalyzer = cms.EDAnalyzer(
    "ZGlobalVsSAIsolationAnalyzer",
    src = cms.InputTag("goodZToMuMu"),
    isoCut = cms.double(1.7),
    veto = cms.double(0.001),
    ptThreshold = cms.double(1),
    etEcalThreshold = cms.double(0.2),
    etHcalThreshold = cms.double(0.5),
    deltaRTrk = cms.double(0.2),
    deltaREcal = cms.double(0.25),
    deltaRHcal = cms.double(0.25),
    alpha = cms.double(0.75),
    beta = cms.double(-0.75)
    )

process.path = cms.Path(
    process.isoAnalyzer
    )



