import FWCore.ParameterSet.Config as cms

stringResolution = cms.ESProducer("StringResolutionProviderESProducer",
    parametrization = cms.string (''),   # specify parametrization (see SWGuidePATKinematicResolutions#Definitions_of_parametrization for more details) 
    resolutions     = cms.vstring(''),   # define sigmas of each parameter (depending on the choice of parametrization), one per string
    constraints     = cms.vdouble(0),    # add constraints (depending on the choice of parametrization)
)
