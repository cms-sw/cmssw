from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF
import RecoTracker.FinalTrackSelectors.TrackTfClassifier_cfi as _mod

trackTfClassifier = _mod.TrackTfClassifier.clone()
trackdnn_CKF.toModify(trackTfClassifier.mva, tfDnnLabel = 'trackSelectionTf_CKF')
