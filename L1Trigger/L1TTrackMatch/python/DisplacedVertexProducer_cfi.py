import FWCore.ParameterSet.Config as cms

DisplacedVertexProducer = cms.EDProducer('DisplacedVertexProducer',
  l1TracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForDispVert", "Level1TTTracksExtendedSelected"),
  l1TrackVertexCollectionName = cms.string("dispVertices"),
  mcTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigisExtended", "Level1TTTracks"),
  ONNXmodel = cms.string("/afs/cern.ch/user/r/rmccarth/private/dispVert/l1tOfflinePR/CMSSW_14_0_0_pre3/src/L1Trigger/L1TTrackMatch/test/dispVertSlim_model.onnx"),
  ONNXInputName = cms.string("feature_input")
)

'''
Features for displaced vertex BDT: ['trkExt_pt_firstTrk', 'trkExt_pt', 'trkExt_eta_firstTrk', 'trkExt_eta', 'trkExt_phi_firstTrk', 'trkExt_phi', 'trkExt_d0_firstTrk', 'trkExt_d0', 'trkExt_z0_firstTrk', 'trkExt_z0', 'trkExt_chi2rz_firstTrk', 'trkExt_chi2rz', 'trkExt_bendchi2_firstTrk', 'trkExt_bendchi2', 'trkExt_MVA_firstTrk', 'trkExt_MVA', 'trkExt_MVA2_firstTrk', 'trkExt_MVA2', 'dv_d_T', 'dv_R_T', 'dv_cos_T', 'dv_del_Z'])

dv inputs are vertex quantities and trkExt is a displaced track property. The firstTrk suffix means the track quantity comes from the higher pt track associated to a vertex. If there's no firstTrk suffix, then the track property is from the lower pt track associated to a vertex.
'''
