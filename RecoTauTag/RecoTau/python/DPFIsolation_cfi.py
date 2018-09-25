
import FWCore.ParameterSet.Config as cms

DPFIsolation = cms.EDProducer("DPFIsolation",
    pfcands     = cms.InputTag('packedPFCandidates'),
    taus 	= cms.InputTag('slimmedTaus'),
    vertices    = cms.InputTag('offlineSlimmedPrimaryVertices'),
    graph_file  = cms.string('RecoTauTag/RecoTau/data/DPFIsolation_2017v0.pb')
)

