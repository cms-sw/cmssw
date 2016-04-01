import FWCore.ParameterSet.Config as cms

hiFJGridEmptyAreaCalculator = cms.EDProducer('HiFJGridEmptyAreaCalculator',
                                 gridWidth = cms.untracked.double(0.05),
                                 bandWidth = cms.untracked.double(0.2),
                                 mapEtaEdges = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
                                 mapToRho = cms.InputTag('hiFJRhoProducer','mapToRho'),
                                 mapToRhoM = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
                                 pfCandSource = cms.InputTag('particleFlowTmp'),
                                 jetSource = cms.InputTag('kt4PFJets'),
								 doCentrality = cms.untracked.bool(True),
								 hiBinCut = cms.untracked.int32(100),    
								 CentralityBinSrc = cms.InputTag("centralityBin","HFtowers"),
)

