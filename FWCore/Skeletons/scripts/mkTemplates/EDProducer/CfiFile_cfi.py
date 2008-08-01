import FWCore.ParameterSet.Config as cms

demo = cms.EDProducer('prodname'
@example_myparticle     , muons = cms.untracked.InputTag('muons')
@example_myparticle     , electrons = cms.untracked.InputTag('pixelMatchGsfElectrons')
)
