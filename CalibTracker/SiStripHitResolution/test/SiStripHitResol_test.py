import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register("isUnitTest",
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "are we running the unit test")
options.parseArguments()

process = cms.Process("HitResol")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')  

InputTagName = "ALCARECOSiStripCalMinBias"
OutputRootFile = "hitresol_ALCARECO_2022F.root"
fileNames=cms.untracked.vstring("/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/ef6009e4-6857-40a1-9a55-0c702021caad.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/ef6cdbda-400c-4813-b4c7-9dfacd070e08.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f1a88b5f-8573-403e-aa35-0ad6b57125c0.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f1c537d0-2265-403b-84f9-dacb5a63c03f.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f30d1b57-eda6-4836-9ed6-cd683945a1e0.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f3c7f61d-7f6b-4021-b6c2-a15b66e3f375.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f3e11a67-7a78-4f6e-9b9b-b7687ce16c68.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f4250ffa-e73e-4e42-baa7-aebd8b169105.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f4c4cb8e-5c92-49b4-9fd3-40e09c4cf48a.root",
                                "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f4f5e6bd-0a16-4937-a3eb-5b76333d9c4d.root")

process.source = cms.Source("PoolSource", fileNames=fileNames)

if(options.isUnitTest):
    process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))
else:
    process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000))

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
#process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.refitTracks = process.TrackRefitterP5.clone(src=cms.InputTag(InputTagName))
process.load("CalibTracker.SiStripHitResolution.SiStripHitResol_cff")
process.anResol.combinatorialTracks = cms.InputTag("refitTracks")
process.anResol.trajectories = cms.InputTag("refitTracks")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(OutputRootFile)  
                                   )

process.allPath = cms.Path(process.MeasurementTrackerEvent*process.offlineBeamSpot*process.refitTracks*process.hitresol)
