import FWCore.ParameterSet.Config as cms

DisplacedVertexProducer = cms.EDProducer('DisplacedVertexProducer',
  l1TracksInputTag = cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),
  l1TrackVertexCollectionName = cms.string("dispVertices"),
  mcTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigisExtended", "Level1TTTracks"),
  ONNXmodel = cms.string("/afs/cern.ch/user/r/rmccarth/private/dispVert/CMSSW_14_0_0_pre3/src/L1Trigger/L1TTrackMatch/test/trackAndVert_D95LowFake_model.onnx"),
  ONNXInputName = cms.string("feature_input"),
  featureNames = cms.vstring(['trkExt_pt_firstTrk', 'trkExt_pt', 'trkExt_eta_firstTrk', 'trkExt_eta', 'trkExt_phi_firstTrk', 'trkExt_phi', 'trkExt_d0_firstTrk', 'trkExt_d0', 'trkExt_z0_firstTrk', 'trkExt_z0', 'trkExt_chi2rz_firstTrk', 'trkExt_chi2rz', 'trkExt_bendchi2_firstTrk', 'trkExt_bendchi2', 'trkExt_MVA_firstTrk', 'trkExt_MVA', 'trkExt_MVA2_firstTrk', 'trkExt_MVA2', 'dv_d_T', 'dv_R_T', 'dv_cos_T', 'dv_del_Z'])
)
