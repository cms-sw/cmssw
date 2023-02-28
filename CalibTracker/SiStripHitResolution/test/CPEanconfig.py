import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register("isUnitTest",
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "are we running the unit test")
options.parseArguments()

process = cms.Process("CPEana")

### Standard Configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

### global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')

### initialize MessageLogger and output report
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripCPEAnalyzer =dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32((10 if options.isUnitTest else 100 ))
                                   ),                                                      
    SiStripCPEAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

### Events and data source

if(options.isUnitTest):
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
else:
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/ef6009e4-6857-40a1-9a55-0c702021caad.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/ef6cdbda-400c-4813-b4c7-9dfacd070e08.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f1a88b5f-8573-403e-aa35-0ad6b57125c0.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f1c537d0-2265-403b-84f9-dacb5a63c03f.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f30d1b57-eda6-4836-9ed6-cd683945a1e0.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f3c7f61d-7f6b-4021-b6c2-a15b66e3f375.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f3e11a67-7a78-4f6e-9b9b-b7687ce16c68.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f4250ffa-e73e-4e42-baa7-aebd8b169105.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f4c4cb8e-5c92-49b4-9fd3-40e09c4cf48a.root",
    "/store/express/Run2022F/StreamExpress/ALCARECO/SiStripCalMinBias-Express-v1/000/362/167/00000/f4f5e6bd-0a16-4937-a3eb-5b76333d9c4d.root"))

### Track refitter specific stuff
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
import RecoTracker.TrackProducer.TrackRefitter_cfi
import CommonTools.RecoAlgos.recoTrackRefSelector_cfi
process.mytkselector = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
process.mytkselector.src = 'ALCARECOSiStripCalMinBias'
process.mytkselector.quality = ['highPurity']
process.mytkselector.min3DLayer = 2
process.mytkselector.ptMin = 0.5
process.mytkselector.tip = 1.0
process.myRefittedTracks = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.myRefittedTracks.src= 'mytkselector'
process.myRefittedTracks.NavigationSchool = ''
process.myRefittedTracks.Fitter = 'FlexibleKFFittingSmoother'

### Analyzer
process.SiStripCPEAnalyzer = cms.EDAnalyzer('SiStripCPEAnalyzer',
                                            tracks = cms.untracked.InputTag("ALCARECOSiStripCalMinBias",""),
                                            trajectories = cms.untracked.InputTag('myRefittedTracks'),
                                            association = cms.untracked.InputTag('myRefittedTracks'),
                                            clusters = cms.untracked.InputTag('ALCARECOSiStripCalMinBias'),
                                            StripCPE = cms.ESInputTag('StripCPEfromTrackAngleESProducer:StripCPEfromTrackAngle'))

### TFileService: output histogram or ntuple
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('histodemo.root'))

### Finally, put together the sequence
process.p = cms.Path(process.offlineBeamSpot*process.mytkselector+process.myRefittedTracks+process.SiStripCPEAnalyzer)
