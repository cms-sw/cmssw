import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigTreeGen")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False


process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")
process.load("L1Trigger.DTTrigger.dttriganalyzer_cfi")
process.dttriganalyzer.debug = True

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:digi.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.p = cms.Path(process.dttriganalyzer)

