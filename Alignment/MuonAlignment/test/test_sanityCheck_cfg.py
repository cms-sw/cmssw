import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_36Y_V10::All"
process.load("Configuration.StandardSequences.Geometry_cff")

from Alignment.MuonAlignment.MuonGeometrySanityCheck_cfi import *
process.MuonGeometrySanityCheck = MuonGeometrySanityCheck.clone()

for detector in detectors(dt=True, csc=True, me42=False, chambers=True, superlayers=True, layers=True):
    process.MuonGeometrySanityCheck.points.append(cms.PSet(
        name = cms.string(detector),
        detector = cms.string(detector),
        frame = cms.string("global"),
        displacement = cms.vdouble(0, 0, 0),
        expectation = cms.vdouble(0, 0, 0),
        outputFrame = cms.string("chamber")))

process.Path = cms.Path(process.MuonGeometrySanityCheck)
