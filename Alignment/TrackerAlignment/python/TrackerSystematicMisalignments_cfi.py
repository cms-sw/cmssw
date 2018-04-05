import FWCore.ParameterSet.Config as cms

TrackerSystematicMisalignments = cms.EDAnalyzer("TrackerSystematicMisalignments",
    # grab an existing geometry
    fromDBGeom = cms.untracked.bool(True),
    #epsilons
    radialEpsilon = cms.untracked.double(-999.0), # default 5e-4 ~ 600 um
    telescopeEpsilon = cms.untracked.double(-999.0), # default 5e-4 ~ 600 um
    layerRotEpsilon = cms.untracked.double(-999.0), # 9.43e-6                   cm^-1
    bowingEpsilon = cms.untracked.double(-999.0), #6.77e-9                      cm^-2
    zExpEpsilon = cms.untracked.double(-999.0), # 2.02e-4
    twistEpsilon = cms.untracked.double(-999.0),# 2.04e-6                       cm^-1
    ellipticalEpsilon = cms.untracked.double(-999.0), # 5e-4
    skewEpsilon = cms.untracked.double(-999.0), # 5.5e-2                        cm
    sagittaEpsilon = cms.untracked.double(-999.0), #5.0e-4

    #misalignment phases
    #0 <= delta < 2pi, epsilon >= 0 for unique results
    #delta=0 reproduces the old behavior

    ellipticalDelta = cms.untracked.double(0),
    skewDelta = cms.untracked.double(0),
    sagittaDelta = cms.untracked.double(0),

    # suppress blind movements
    suppressBlindMvmts = cms.untracked.bool(False),
	# compatibility flag for old z convention
	oldMinusZconvention = cms.untracked.bool(False)
)

