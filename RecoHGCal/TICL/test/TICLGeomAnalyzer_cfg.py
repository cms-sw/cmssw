import importlib

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('geometry',
                 'Run4D121',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 'Geometry to test: Run4D121 or Run4D122')
options.parseArguments()

# geometry -> (geometry cff, era, GlobalTag), following upgradeWorkflowComponents
GEOMETRIES = {
    'Run4D121': ('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff', 'Phase2C22I13M9',
                 'auto:phase2_realistic_T35'),
    'Run4D122': ('Configuration.Geometry.GeometryExtendedRun4D122Reco_cff', 'Phase2C26I13M9',
                 'auto:phase2_realistic_T35'),
}
geometryCff, eraName, globalTag = GEOMETRIES[options.geometry]
era = getattr(importlib.import_module('Configuration.Eras.Era_' + eraName + '_cff'), eraName)

process = cms.Process("TICLGeomAnalyze", era)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load(geometryCff)
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Accelerators_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

# Default HGCal geometry (label "") plus the opt-in barrel instances
process.load("RecoHGCal.TICL.TICLGeom_cff")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")

# Host-side closure tests against RecHitTools
process.ticlGeomAnalyzerHGCal = cms.EDAnalyzer("TICLGeomAnalyzer", label = cms.string(""))
process.ticlGeomAnalyzerECAL = cms.EDAnalyzer("TICLGeomAnalyzer", label = cms.string("ECAL"))
process.ticlGeomAnalyzerHCAL = cms.EDAnalyzer("TICLGeomAnalyzer", label = cms.string("HCAL"))

# Device-side lookup test on the alpaka backend (automatic host to device copy)
process.ticlGeomDeviceTest = cms.EDProducer("TICLGeomDeviceTest@alpaka",
    src = cms.ESInputTag("", "")
)

process.p = cms.Path(
    process.ticlGeomAnalyzerHGCal +
    process.ticlGeomAnalyzerECAL +
    process.ticlGeomAnalyzerHCAL +
    process.ticlGeomDeviceTest
)
