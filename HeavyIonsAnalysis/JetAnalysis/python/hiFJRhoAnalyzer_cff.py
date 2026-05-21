import FWCore.ParameterSet.Config as cms

hiFJRhoAnalyzer = cms.EDAnalyzer(
    'HiFJRhoAnalyzer',
    etaMap        = cms.InputTag('hiFJRhoProducer','mapEtaEdges','HiForest'),
    rho           = cms.InputTag('hiFJRhoProducer','mapToRho'),
    rhom          = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
    rhoCorr       = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoCorr'),
    rhomCorr      = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoMCorr'),
    rhoCorr1Bin   = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoCorr1Bin'),
    rhomCorr1Bin  = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoMCorr1Bin'),
    rhoGrid       = cms.InputTag('hiFJGridEmptyAreaCalculator','mapRhoVsEtaGrid'),
    meanRhoGrid   = cms.InputTag('hiFJGridEmptyAreaCalculator','mapMeanRhoVsEtaGrid'),
    etaMaxRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculator','mapEtaMaxGrid'),
    etaMinRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculator','mapEtaMinGrid'),
    ptJets        = cms.InputTag('hiFJRhoProducer','ptJets'),
    etaJets       = cms.InputTag('hiFJRhoProducer','etaJets'),
    areaJets      = cms.InputTag('hiFJRhoProducer','areaJets'),
    useModulatedRho = cms.bool(False),
)

hiFJRhoAnalyzerFinerBins = cms.EDAnalyzer(
    'HiFJRhoAnalyzer',
    etaMap        = cms.InputTag('hiFJRhoProducerFinerBins','mapEtaEdges','HiForest'),
    rho           = cms.InputTag('hiFJRhoProducerFinerBins','mapToRho'),
    rhom          = cms.InputTag('hiFJRhoProducerFinerBins','mapToRhoM'),
    rhoCorr       = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapToRhoCorr'),
    rhomCorr      = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapToRhoMCorr'),
    rhoCorr1Bin   = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapToRhoCorr1Bin'),
    rhomCorr1Bin  = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapToRhoMCorr1Bin'),
    rhoGrid       = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapRhoVsEtaGrid'),
    meanRhoGrid   = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapMeanRhoVsEtaGrid'),
    etaMaxRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapEtaMaxGrid'),
    etaMinRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculatorFinerBins','mapEtaMinGrid'),
    ptJets        = cms.InputTag('hiFJRhoProducerFinerBins','ptJets'),
    etaJets       = cms.InputTag('hiFJRhoProducerFinerBins','etaJets'),
    areaJets      = cms.InputTag('hiFJRhoProducerFinerBins','areaJets'),
    useModulatedRho = cms.bool(False),
)

hiPuRhoR3Analyzer = hiFJRhoAnalyzer.clone(
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
    rhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculator','mapRhoVsEtaGrid'),
    meanRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculator','mapMeanRhoVsEtaGrid'),
    etaMaxRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculator','mapEtaMaxGrid'),
    etaMinRhoGrid = cms.InputTag('hiFJGridEmptyAreaCalculator','mapEtaMinGrid'),
    rhoFlowFitParams = cms.InputTag('hiFJRhoFlowModulationProducer','rhoFlowFitParams'),
    ptJets = cms.InputTag('hiPuRhoR3Producer', 'ptJets'),
    etaJets = cms.InputTag('hiPuRhoR3Producer', 'etaJets'),
    areaJets = cms.InputTag('hiPuRhoR3Producer', 'areaJets'),
    useModulatedRho = cms.bool(True),
)

# Add rho estimator
from RecoHI.HiJetAlgos.HiRecoPFJets_cff import kt4PFJetsForRho
from RecoHI.HiJetAlgos.hiFJRhoProducer import hiFJRhoProducer
from RecoHI.HiJetAlgos.hiFJGridEmptyAreaCalculator_cff import hiFJGridEmptyAreaCalculator
hiFJRhoProducerFinerBins = hiFJRhoProducer.clone(etaRanges = [-5., -4., -3, -2.5, -2.0, -0.8, 0.8, 2.0, 2.5, 3., 4., 5.])
hiFJGridEmptyAreaCalculatorFinerBins = hiFJGridEmptyAreaCalculator.clone(mapEtaEdges = 'hiFJRhoProducerFinerBins:mapEtaEdges', mapToRho = 'hiFJRhoProducerFinerBins:mapToRho', mapToRhoM = 'hiFJRhoProducerFinerBins:mapToRhoM')
kt4PFJetsForRho.src = 'packedPFCandidates'
hiFJGridEmptyAreaCalculatorFinerBins.pfCandSource = 'packedPFCandidates'
rhoSequence = cms.Sequence(kt4PFJetsForRho + hiFJRhoProducerFinerBins + hiFJGridEmptyAreaCalculatorFinerBins + hiFJRhoAnalyzerFinerBins)
