import FWCore.ParameterSet.Config as cms

hiFJGridEmptyAreaCalculator = cms.EDProducer('HiFJGridEmptyAreaCalculator',
                                             gridWidth = cms.double(0.05),
                                             bandWidth = cms.double(0.2),
                                             mapEtaEdges = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
                                             mapToRho = cms.InputTag('hiFJRhoProducer','mapToRho'),
                                             mapToRhoM = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
                                             pfCandSource = cms.InputTag('particleFlowTmp'),
                                             jetSource = cms.InputTag('kt4PFJetsForRho'),
					     doCentrality = cms.bool(True),
					     hiBinCut = cms.int32(100),    
					     CentralityBinSrc = cms.InputTag("centralityBin","HFtowers"),
					     keepGridInfo = cms.bool(False),
)

