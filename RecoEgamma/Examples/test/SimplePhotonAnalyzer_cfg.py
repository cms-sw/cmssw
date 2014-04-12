import FWCore.ParameterSet.Config as cms

process = cms.Process("simplePhotonAnalyzer")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoEgamma.Examples.simplePhotonAnalyzer_cfi")

process.GlobalTag.globaltag = 'MC_31X_V9::All'


process.maxEvents = cms.untracked.PSet(
 input = cms.untracked.int32(3000)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('simplePhotonAnalysis.root')

)

from RecoEgamma.Examples.simplePhotonAnalyzer_cfi import *
simplePhotonAnalyzer.sample=3


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

# official RelVal 334 Single Gamma Pt35 
 #   '/store/relval/CMSSW_3_3_4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V9-v1/0001/BC77B293-6BD5-DE11-9ADE-0018F3D09660.root',
 #   '/store/relval/CMSSW_3_3_4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V9-v1/0000/023E8FA1-29D5-DE11-96B3-003048678F74.root'


# official RelVal 334 H130GGgluonfusio

    '/store/relval/CMSSW_3_3_4/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8A-v1/0001/F4AEE6DD-64D5-DE11-B4DE-002618943843.root',
    '/store/relval/CMSSW_3_3_4/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8A-v1/0000/7C10B2AF-38D5-DE11-8A42-003048678ADA.root',
    '/store/relval/CMSSW_3_3_4/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8A-v1/0000/7A92493F-37D5-DE11-8EAE-001731AF68B9.root',
    '/store/relval/CMSSW_3_3_4/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8A-v1/0000/6026AF77-38D5-DE11-987B-0017319C95D6.root',
    '/store/relval/CMSSW_3_3_4/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8A-v1/0000/563B7851-3AD5-DE11-A44E-0018F3D0963C.root',
    '/store/relval/CMSSW_3_3_4/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8A-v1/0000/2863B76F-38D5-DE11-9817-0018F3D0968C.root'


    )
)



process.p1 = cms.Path(process.simplePhotonAnalyzer)
process.schedule = cms.Schedule(process.p1)



