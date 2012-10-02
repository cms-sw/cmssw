import FWCore.ParameterSet.Config as cms

testanalyzer = cms.EDAnalyzer("GSRecHitValidation",
                              matchedHitCollectionInputTag = cms.InputTag("siTrackerGaussianSmearingRecHits", "TrackerGSMatchedRecHits"),
                              hitCollectionInputTag = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits"),
                              #TID
                              TID_Pos_x_AxisLim = cms.double(6.0),
                              TID_Pos_y_AxisLim = cms.double(10.0),
                              TID_Res_x_AxisLim = cms.double(0.02),
                              TID_Res_y_AxisLim = cms.double(0.5),
                              TID_Pull_x_AxisLim = cms.double(5.0),
                              TID_Pull_y_AxisLim = cms.double(5.0),
                              
                              #TEC
                              TEC_Pos_x_AxisLim = cms.double(6.0),
                              TEC_Pos_y_AxisLim = cms.double(10.0),
                              TEC_Res_x_AxisLim = cms.double(0.02),
                              TEC_Res_y_AxisLim = cms.double(0.5),
                              TEC_Pull_x_AxisLim = cms.double(5.0),
                              TEC_Pull_y_AxisLim = cms.double(5.0),
                              
                              #TIB
                              TIB_Pos_x_AxisLim = cms.double(6.0),
                              TIB_Pos_y_AxisLim = cms.double(10.0),
                              TIB_Res_x_AxisLim = cms.double(0.02),
                              TIB_Res_y_AxisLim = cms.double(0.5),
                              TIB_Pull_x_AxisLim = cms.double(5.0),
                              TIB_Pull_y_AxisLim = cms.double(5.0),
                              
                              #TOB
                              TOB_Pos_x_AxisLim = cms.double(6.0),
                              TOB_Pos_y_AxisLim = cms.double(10.0),
                              TOB_Res_x_AxisLim = cms.double(0.02),
                              TOB_Res_y_AxisLim = cms.double(0.5),
                              TOB_Pull_x_AxisLim = cms.double(5.0),
                              TOB_Pull_y_AxisLim = cms.double(5.0),
                              
                              #PXB
                              PXB_SimPos_AxisLim = cms.double(1.0),
                              PXB_RecPos_AxisLim = cms.double(1.0),
                              PXB_Res_AxisLim = cms.double(0.01),
                              PXB_Err_AxisLim = cms.double(0.0),
                              
                              #PXF
                              PXF_SimPos_AxisLim = cms.double(1.0),
                              PXF_RecPos_AxisLim = cms.double(1.0),
                              PXF_Res_AxisLim = cms.double(0.01),
                              PXF_Err_AxisLim = cms.double(0.0),
                              
                              #File stuff
                              outfilename = cms.string('Your_Output_File.root'),
                              TrackProducer = cms.string('generalTracks'),
                              SimHitList = cms.vstring('famosSimHitsTrackerHits')
                              )



