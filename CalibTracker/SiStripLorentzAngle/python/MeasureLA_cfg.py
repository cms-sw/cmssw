import FWCore.ParameterSet.Config as cms

process = cms.Process("MACRO")
process.add_(cms.Service("MessageLogger"))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("CalibTracker.SiStripLorentzAngle.MeasureLA_cfi")
process.measureLA.InputFiles += []

