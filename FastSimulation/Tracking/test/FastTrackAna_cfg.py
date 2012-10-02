import FWCore.ParameterSet.Config as cms

process = cms.Process("ana")
# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("FastSimulation.Configuration.CommonInputsFake_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:tracks_Gun_test.root')
)

process.testanalyzer = cms.EDAnalyzer("FastTrackAnalyzer",
    TID_Erry_AxisLim = cms.double(0.1),
    TID_Err_AxisLim = cms.double(0.02),
    TIB_Resy_AxisLim = cms.double(0.5),
    PXB_SimPos_AxisLim = cms.double(1.0),
    TOB_Erry_AxisLim = cms.double(0.1),
    PXF_SimPos_AxisLim = cms.double(1.0),
    TEC_Pos_AxisLim = cms.double(10.0),
    TIB_Pos_AxisLim = cms.double(10.0),
    TIB_Erry_AxisLim = cms.double(0.1),
    TEC_Err_AxisLim = cms.double(0.02),
    PXF_Err_AxisLim = cms.double(0.0),
    TOB_Err_AxisLim = cms.double(0.01),
    TIB_Res_AxisLim = cms.double(0.01),
    outfilename = cms.string('ResHistos.root'),
    TOB_Res_AxisLim = cms.double(0.05),
    TID_Resy_AxisLim = cms.double(0.5),
    PXB_RecPos_AxisLim = cms.double(1.0),
    PXB_Res_AxisLim = cms.double(0.01),
    PXF_RecPos_AxisLim = cms.double(1.0),
    PXB_Err_AxisLim = cms.double(0.0),
    TEC_Res_AxisLim = cms.double(0.05),
    TOB_Resy_AxisLim = cms.double(0.5),
    TID_Pos_AxisLim = cms.double(10.0),
    TID_Res_AxisLim = cms.double(0.05),
    TEC_Resy_AxisLim = cms.double(0.5),
    TrackProducer = cms.string('generalTracks'),
    PXF_Res_AxisLim = cms.double(0.01),
    #increase for y errors
    TIB_Err_AxisLim = cms.double(0.01),
    NumTracks_AxisLim = cms.int32(10),
    TEC_Erry_AxisLim = cms.double(0.1),
    TOB_Pos_AxisLim = cms.double(10.0),
    SimHitList = cms.vstring('famosSimHitsTrackerHits'),
    simVertexContainerTag = cms.InputTag('famosSimHits'),
    siTrackerGSRecHit2DCollectionTag = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits")
)

process.p = cms.Path(process.testanalyzer)
process.MessageLogger.destinations = ['detailedInfo.txt']


