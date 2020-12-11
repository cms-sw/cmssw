import FWCore.ParameterSet.Config as cms

process = cms.Process("demo")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryExtended2018Reco_cff")
#process.load("Configuration.Geometry.GeometryExtended2026D49Reco_cff")
process.load("CalibCalorimetry.HcalPlugins.HBHEDarkening_cff")

import CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff

process.ana = cms.EDAnalyzer("HBHEDarkeningAnalyzer",
    deliveredLumi = cms.double(304),
    HBmeanenergies = CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff.es_hardcode.HBmeanenergies,
    HEmeanenergies = CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff.es_hardcode.HEmeanenergies,
)

process.p1 = cms.Path(process.ana)
