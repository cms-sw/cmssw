import FWCore.ParameterSet.Config as cms

demo = cms.EDProducer('__class__'
@example_myparticle     , muons = cms.InputTag('muons')
@example_myparticle     , electrons = cms.InputTag('pixelMatchGsfElectrons')
)
