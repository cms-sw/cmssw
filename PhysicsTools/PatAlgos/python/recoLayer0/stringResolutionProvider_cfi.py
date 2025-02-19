import FWCore.ParameterSet.Config as cms

stringResolution = cms.ESProducer("StringResolutionProviderESProducer",
  ## specify parametrization (see
  ## SWGuidePATKinematicResolutions for more details)
  parametrization = cms.string ('EtEtaPhi'),
  functions = cms.VPSet(
    cms.PSet(
      ## set the eta bin as selection string.(optional)
      ## See SWGuidePhysicsCutParser for more details
      bin = cms.string(""),
      ## define resolution functions of each parameter
      et  = cms.string("et * (sqrt(0.08^2 + (1./sqrt(et))^2 + (5./et)^2))"),
      eta = cms.string("sqrt(0.008^2 + (1.5/et)^2)"),
      phi = cms.string("sqrt(0.008^2 + (2.6/et)^2)"),
    ),
  ),
  ## add constraints (depending on the choice of para-
  ## metrization); for et/eta/phi this has to be set
  ## to 0 (have a look at SWGuidePATKinematicResolutions
  ## for more details)
  constraints = cms.vdouble(0)
)
