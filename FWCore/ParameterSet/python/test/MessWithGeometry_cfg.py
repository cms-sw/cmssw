import FWCore.ParameterSet.Config as cms

process = cms.Process("P")
process.setStrict(True)
process.load("FWCore.ParameterSet.test.Geometry_cfi")
process.load("FWCore.ParameterSet.test.MessWithGeometry_cff")
process.load("FWCore.ParameterSet.test.MessWithPreshower_cff")
#check that both changes got merged
print process.geometry.dumpPython()

