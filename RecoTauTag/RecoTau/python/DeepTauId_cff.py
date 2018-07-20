
import FWCore.ParameterSet.Config as cms

deepTauIdraw = cms.EDProducer("DeepTauId",
    electrons = cms.InputTag('slimmedElectrons'),
    muons = cms.InputTag('slimmedMuons'),
    taus = cms.InputTag('slimmedTaus'),
    graph_file = cms.string('RecoTauTag/RecoTau/data/deepTau_2017v1_20L1024N.pb')
)

deepTauIdSeq = cms.Sequence(deepTauIdraw)
