import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_36Y_V10::All"
process.load("Configuration.StandardSequences.Geometry_cff")

from Alignment.MuonAlignment.MuonGeometrySanityCheck_cfi import *
process.MuonGeometrySanityCheck = MuonGeometrySanityCheck.clone()

from math import sin, cos
process.MuonGeometrySanityCheck.frames.append(cms.PSet(
        name = cms.string("custom1"),
        matrix = cms.vdouble(cos(0.1), -sin(0.1), 0,
                             sin(0.1),  cos(0.1), 0,
                                    0,         0, 1)))

process.MuonGeometrySanityCheck.points.append(cms.PSet(
        name = cms.string("first"),
        detector = cms.string("ME+2/2/30"),
        frame = cms.string("custom1"),
        displacement = cms.vdouble(0., 120., 10.),
        expectation = cms.vdouble(180.073603271, -494.748183105, 816.161244141),
        outputFrame = cms.string("global")))

process.MuonGeometrySanityCheck.points.append(cms.PSet(
        name = cms.string("first"),
        detector = cms.string("ME+2/2/31"),
        frame = cms.string("custom1"),
        displacement = cms.vdouble(0., 120., 10.),
        expectation = cms.vdouble(263.25, -455.962366211, 840.961244141),
        outputFrame = cms.string("global")))


# for detector in detectors(dt=True, csc=True, me42=False, chambers=True, superlayers=True, layers=True):
#     process.MuonGeometrySanityCheck.points.append(cms.PSet(
#         name = cms.string(detector),
#         detector = cms.string(detector),
#         frame = cms.string("global"),
#         displacement = cms.vdouble(0, 0, 0),
#         expectation = cms.vdouble(0, 0, 0),
#         outputFrame = cms.string("chamber")))

process.Path = cms.Path(process.MuonGeometrySanityCheck)
