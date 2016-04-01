import FWCore.ParameterSet.Config as cms

hiFJRhoAnalyzer = cms.EDAnalyzer('HiFJRhoAnalyzer',
                                 etaMap             = cms.InputTag('hiFJRhoProducer','mapEtaEdges','HiForest'),
                                 rho                = cms.InputTag('hiFJRhoProducer','mapToRho'),
                                 rhom               = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
                                 rhoCorr            = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoCorr'),
                                 rhomCorr           = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoMCorr'),
                                 rhoCorr1Bin            = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoCorr1Bin'),
                                 rhomCorr1Bin           = cms.InputTag('hiFJGridEmptyAreaCalculator','mapToRhoMCorr1Bin'),
                                 rhoGrid            = cms.InputTag('hiFJGridEmptyAreaCalculator','mapRhoVsEtaGrid'),
                                 meanRhoGrid            = cms.InputTag('hiFJGridEmptyAreaCalculator','mapMeanRhoVsEtaGrid'),
                                 etaMaxRhoGrid            = cms.InputTag('hiFJGridEmptyAreaCalculator','mapEtaMaxGrid'),
                                 etaMinRhoGrid            = cms.InputTag('hiFJGridEmptyAreaCalculator','mapEtaMinGrid'),
                                 ptJets            = cms.InputTag('hiFJRhoProducer','ptJets'),
                                 etaJets            = cms.InputTag('hiFJRhoProducer','etaJets'),
                                 areaJets            = cms.InputTag('hiFJRhoProducer','areaJets'),
)

