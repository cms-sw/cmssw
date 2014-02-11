import FWCore.ParameterSet.Config as cms

#needed for MLP tagger
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
goodOfflinePrimaryVerticesQG = cms.EDFilter("PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src          = cms.InputTag('offlinePrimaryVertices')
)

from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets
kt6PFJetsQG = kt6PFJets.clone()

#needed for Likelihood tagger
from RecoJets.JetProducers.kt4PFJets_cfi import *
kt6PFJetsIsoQG = kt4PFJets.clone( rParam = 0.6, doRhoFastjet = True )
kt6PFJetsIsoQG.Rho_EtaMax = cms.double(2.5)


QGTagger = cms.EDProducer('QGTagger',
  srcRho          = cms.InputTag('kt6PFJetsQG','rho'),
  srcRhoIso       = cms.InputTag('kt6PFJetsIsoQG','rho'),
)

#QuarkGluonTagger = cms.Sequence(goodOfflinePrimaryVerticesQG + kt6PFJetsQG + kt6PFJetsIsoQG + QGTagger)
