import FWCore.ParameterSet.Config as cms

stringResolution = cms.ESProducer("StringResolutionProviderESProducer",
  ## specify parametrization (see
  ## SWGuidePATKinematicResolutions for more details)
  parametrization = cms.string (''),
  functions = cms.VPSet(
    cms.PSet(
      ## set the eta bin as selection string. See
      ## SWGuidePhysicsCutParser for more details
      bin = cms.string(""),
      ## define resolution functions of each para-
      ## meter
      et  = cms.string(""),
      eta = cms.string(""),
      phi = cms.string(""),
    ),
  ),
  ## add constraints (depending on the choice of
  ## parametrization); for et/eta/phi this has to
  ## be set to 0 (have a look at SWGuidePATKinematicResolutions
  ## for more details)
  constraints = cms.vdouble(0)
)
