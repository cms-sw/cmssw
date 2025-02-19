import FWCore.ParameterSet.Config as cms

demo = cms.EDProducer('prodname'
@example_myparticle     , muons = cms.InputTag('muons')
@example_myparticle     , electrons = cms.InputTag('pixelMatchGsfElectrons')
)
