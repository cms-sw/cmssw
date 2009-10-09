import FWCore.ParameterSet.Config as cms

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *

#CalibrationTracksRefit = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(src = cms.InputTag("CalibrationTracks"))
CalibrationTracksRefit = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(src = cms.InputTag("CalibrationTracks"))
CalibrationTracks = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'ALCARECOTkAlCosmicsCTF0T',
    filter = True,
    applyBasicCuts = True,
    ptMin = 1.5,
    nHitMin = 4,
    chi2nMax = 10.,
    )
trackFilterRefit = cms.Sequence( CalibrationTracks + offlineBeamSpot + CalibrationTracksRefit )

CalibrationTracksP5 = CalibrationTracks.clone( src = 'ctfWithMaterialTracksP5')
CalibrationTracksRefitP5 =  RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(src = cms.InputTag("CalibrationTracksP5"))
trackFilterRefitP5 = cms.Sequence( CalibrationTracksP5 + offlineBeamSpot + CalibrationTracksRefitP5 )

CalibrationTracksRAW = CalibrationTracks.clone( src = 'generalTracks')
CalibrationTracksRefitRAW =  RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(src = cms.InputTag("CalibrationTracksRAW"))
trackFilterRefitRAW = cms.Sequence( CalibrationTracksRAW + offlineBeamSpot + CalibrationTracksRefitRAW )
