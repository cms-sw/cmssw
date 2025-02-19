import FWCore.ParameterSet.Config as cms

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *

CalibrationTracksRefit = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(src = cms.InputTag("CalibrationTracks"))
CalibrationTracks = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'generalTracks',
    filter = True,
    applyBasicCuts = True,
    ptMin = 0.8,
    nHitMin = 6,
    chi2nMax = 10.,
    )
trackFilterRefit = cms.Sequence( CalibrationTracks + offlineBeamSpot + CalibrationTracksRefit )

CalibrationTracksRAW = CalibrationTracks.clone()
CalibrationTracksRefitRAW =  RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(src = cms.InputTag("CalibrationTracksRAW"))
trackFilterRefitRAW = cms.Sequence( CalibrationTracksRAW + offlineBeamSpot + CalibrationTracksRefitRAW )

CalibrationTracksP5 = CalibrationTracks.clone( src = 'ctfWithMaterialTracksP5')
CalibrationTracksRefitP5 =  RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(src = cms.InputTag("CalibrationTracksP5"))
trackFilterRefitP5 = cms.Sequence( CalibrationTracksP5 + offlineBeamSpot + CalibrationTracksRefitP5 )

CalibrationTracksAlcaP5 = CalibrationTracks.clone( src = 'ALCARECOTkAlCosmicsCTF0T')
CalibrationTracksRefitAlcaP5 =  RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(src = cms.InputTag("CalibrationTracksAlcaP5"))
trackFilterRefitAlcaP5 = cms.Sequence( CalibrationTracksAlcaP5 + offlineBeamSpot + CalibrationTracksRefitAlcaP5 )

CalibrationTracksAlca = CalibrationTracks.clone( src = 'ALCARECOTkAlCosmicsCTF0T')
CalibrationTracksRefitAlca =  RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(src = cms.InputTag("CalibrationTracksAlca"))
trackFilterRefitAlca = cms.Sequence( CalibrationTracksAlca + offlineBeamSpot + CalibrationTracksRefitAlca )

