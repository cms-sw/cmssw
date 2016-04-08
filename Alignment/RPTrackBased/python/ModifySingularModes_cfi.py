import FWCore.ParameterSet.Config as cms

ModifySingularModes = cms.EDAnalyzer("ModifySingularModes",
    z0 = cms.untracked.double(0.0),

    inputFile = cms.untracked.string(""),
    outputFile = cms.untracked.string(""),

    # mm
    z1 = cms.untracked.double(0),
    z2 = cms.untracked.double(0),

    # mm
    de_x1 = cms.untracked.double(0),
    de_y1 = cms.untracked.double(0),
    de_x2 = cms.untracked.double(0),
    de_y2 = cms.untracked.double(0),

    # rad
    de_rho1 = cms.untracked.double(0),
    de_rho2 = cms.untracked.double(0),
)
