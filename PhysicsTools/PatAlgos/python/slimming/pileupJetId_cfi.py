import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PileupJetIDParams_cfi import full_5x_chs

pileupJetId = cms.EDProducer('PileupJetIdProducer',
     produceJetIds = cms.bool(True),
     jetids = cms.InputTag(""),
     runMvas = cms.bool(True),
     jets = cms.InputTag("ak5PFJetsCHS"),
     vertexes = cms.InputTag("offlinePrimaryVertices"),
     algos = cms.VPSet(full_5x_chs),
     rho     = cms.InputTag("fixedGridRhoFastjetAll"),
     jec     = cms.string("AK5PFchs"),
     applyJec = cms.bool(True),
     inputIsCorrected = cms.bool(False),
     residualsFromTxt = cms.bool(False),
     residualsTxt     = cms.FileInPath("RecoJets/JetProducers/data/download.url") # must be an existing file
)

