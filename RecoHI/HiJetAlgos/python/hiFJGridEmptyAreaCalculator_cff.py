import FWCore.ParameterSet.Config as cms

hiFJGridEmptyAreaCalculator = cms.EDProducer('HiFJGridEmptyAreaCalculator',
                                             gridWidth = cms.double(0.05),
                                             bandWidth = cms.double(0.2),
                                             mapEtaEdges = cms.InputTag('hiFJRhoProducer','mapEtaEdges'),
                                             mapToRho = cms.InputTag('hiFJRhoProducer','mapToRho'),
                                             mapToRhoM = cms.InputTag('hiFJRhoProducer','mapToRhoM'),
                                             pfCandSource = cms.InputTag('particleFlow'),
                                             jetSource = cms.InputTag('kt4PFJetsForRho'),
					     doCentrality = cms.bool(True),
					     hiBinCut = cms.int32(100),    
					     CentralityBinSrc = cms.InputTag("centralityBin","HFtowers"),
					     keepGridInfo = cms.bool(False),
)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
pA_2016.toModify(hiFJGridEmptyAreaCalculator, doCentrality = False)

