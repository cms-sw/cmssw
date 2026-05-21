import FWCore.ParameterSet.Config as cms

hiPuRhoR3Analyzer = cms.EDAnalyzer('HiFJRhoAnalyzer',
    etaMap = cms.InputTag('hiPuRhoR3Producer','mapEtaEdges','HiForest'),
    rho = cms.InputTag('hiPuRhoR3Producer','mapToRho'),
    rhoExtra = cms.InputTag('hiPuRhoR3Producer','mapToRhoExtra'),
    rhom = cms.InputTag('hiPuRhoR3Producer','mapToRhoM'),
    rhoCorr = cms.InputTag('hiPuRhoR3Producer','mapToRhoMedian'),
    rhomCorr = cms.InputTag('hiPuRhoR3Producer','mapToRhoM'),
    rhoCorr1Bin = cms.InputTag('hiPuRhoR3Producer','mapToRho'),
    rhomCorr1Bin = cms.InputTag('hiPuRhoR3Producer','mapToRhoM'),
    nTow = cms.InputTag('hiPuRhoR3Producer','mapToNTow'),
    towExcludePt = cms.InputTag('hiPuRhoR3Producer','mapToTowExcludePt'),
    towExcludePhi = cms.InputTag('hiPuRhoR3Producer','mapToTowExcludePhi'),
    towExcludeEta = cms.InputTag('hiPuRhoR3Producer','mapToTowExcludeEta'),
    rhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapRhoVsEtaGrid'),
    meanRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapMeanRhoVsEtaGrid'),
    etaMaxRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapEtaMaxGrid'),
    etaMinRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapEtaMinGrid'),
    rhoFlowFitParams = cms.InputTag('hiFJRhoFlowModulation','rhoFlowFitParams'),
    ptJets        = cms.InputTag('hiFJRhoProducerFinerBins','ptJets'),
    etaJets       = cms.InputTag('hiFJRhoProducerFinerBins','etaJets'),
    areaJets      = cms.InputTag('hiFJRhoProducerFinerBins','areaJets'),
    useModulatedRho = cms.bool(True),
)

# Add rho estimator
from HeavyIonsAnalysis.JetAnalysis.extraJets_cff import *
from RecoHI.HiJetAlgos.hiPuRhoProducer_cfi import hiPuRhoProducer as hiPuRhoR3Producer
hiPuRhoR3Producer.src = "PackedPFTowers"
rhoFlowDataSequence = cms.Sequence(extraFlowJetsData + hiPuRhoR3Producer + hiPuRhoR3Analyzer)
rhoFlowMCSequence = cms.Sequence(extraFlowJetsMC + hiPuRhoR3Producer + hiPuRhoR3Analyzer)
