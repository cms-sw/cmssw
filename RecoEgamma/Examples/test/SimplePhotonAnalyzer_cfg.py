import FWCore.ParameterSet.Config as cms

process = cms.Process("simplePhotonAnalyzer")

process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("RecoEgamma.Examples.simplePhotonAnalyzer_cfi")

process.maxEvents = cms.untracked.PSet(
# input = cms.untracked.int32(5000)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('simplePhotonAnalysis.root')

)

from RecoEgamma.Examples.simplePhotonAnalyzer_cfi import *


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

# official RelVal 212 Single Gamma Pt35 
'/store/relval/CMSSW_2_1_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0000/3ACF5E43-8F68-DD11-9995-001617E30F58.root',
'/store/relval/CMSSW_2_1_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0000/4AC0C45E-8F68-DD11-8478-000423D94A04.root',
'/store/relval/CMSSW_2_1_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0000/CE471A58-8F68-DD11-B03B-000423D99896.root',
'/store/relval/CMSSW_2_1_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0000/FA865EF5-9268-DD11-AAFB-000423D6CA6E.root'

)

)



process.p1 = cms.Path(process.simplePhotonAnalyzer)
process.schedule = cms.Schedule(process.p1)



