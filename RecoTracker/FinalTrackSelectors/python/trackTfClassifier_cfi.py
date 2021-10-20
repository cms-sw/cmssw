from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF
import RecoTracker.FinalTrackSelectors.trackTfClassifierDefault_cfi as _mod

trackTfClassifier = _mod.trackTfClassifierDefault.clone()
trackdnn_CKF.toModify(trackTfClassifier.mva, tfDnnLabel = 'trackSelectionTf_CKF')
