import FWCore.ParameterSet.Config as cms

process = cms.Process("newtest")

from Geometry.CaloEventSetup.CaloTopology_cfi import *

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")



process.load("DQM.HLTEvF.HLTMonAlCa_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/user/b/beaucero/scratch0/CMSSW_3_1_2/src/AlCaRawStream1E31.root'
)
)

#process.out1 = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('drop *'),
#    fileName = cms.untracked.string('dqm.root')
#)

process.p = cms.Path(process.HLTAlCaMon*process.MEtoEDMConverter)
#process.o = cms.EndPath(process.out1)
process.EcalPi0Mon.SaveToFile = True
process.EcalPhiSymMon.SaveToFile = True
process.MEtoEDMConverter.Verbosity = 1


