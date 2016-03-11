import FWCore.ParameterSet.Config as cms

hiFJRhoAnalyzer = cms.EDAnalyzer('HiFJRhoAnalyzer',
                                 etaMap    = cms.InputTag('hiFJRhoProducer','mapEtaEdges','HiForest'),
                                 rho       = cms.InputTag('hiFJRhoProducer','mapToRho'),
                                 rhom      = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
)

