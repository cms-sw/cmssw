import FWCore.ParameterSet.Config as cms

ppsFastLocalSimulation = cms.EDProducer("PPSFastLocalSimulation",
    verbosity = cms.untracked.uint32(0),

    makeHepMC = cms.bool(False),
    makeHits = cms.bool(True),
    
    particlesPerEvent = cms.uint32(1),

    particle_p = cms.double(6500),  # in GeV
    particle_E = cms.double(6500),  # in GeV

    z0 = cms.double(214500),

    position_distribution = cms.PSet(
      type = cms.string("box"),
      x_mean = cms.double(5.0),       #in mm
      x_width = cms.double(10.0),
      x_min = cms.double(0.0),
      x_max = cms.double(0.0),

      y_mean = cms.double(0.0),
      y_width = cms.double(20.0),
      y_min = cms.double(0.0),
      y_max = cms.double(0.0)
    ),

    angular_distribution = cms.PSet(
      type = cms.string("gauss"),
      x_mean = cms.double(0.0),       #in rad
      x_width = cms.double(100E-6),
      x_min = cms.double(0E-6),
      x_max = cms.double(0E-6),

      y_mean = cms.double(0.0),
      y_width = cms.double(100E-6),
      y_min = cms.double(0E-6),
      y_max = cms.double(0E-6)
    ),

    #RPs = cms.vuint32(120, 121, 122, 123, 124, 125),
    RPs = cms.vuint32(103, 116, 123),

    roundToPitch = cms.bool(True),

    pitchStrips = cms.double(66E-3),  # mm
    pitchDiamonds = cms.double(200E-3),  # mm
    pitchPixels = cms.double(30E-3),  # mm

    insensitiveMarginStrips = cms.double(34E-3), #mm, save value as RPActiveEdgePosition in SimTotem/RPDigiProducer/python/RPSiDetConf_cfi.py
)
