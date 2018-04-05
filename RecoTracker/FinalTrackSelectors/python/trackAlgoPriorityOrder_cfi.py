from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrderDefault_cfi import trackAlgoPriorityOrderDefault as _trackAlgoPriorityOrderDefault
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

trackAlgoPriorityOrder = _trackAlgoPriorityOrderDefault.clone(
    algoOrder = _cfg.iterationAlgos("")
)

for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toModify(trackAlgoPriorityOrder, algoOrder=_cfg.iterationAlgos(_postfix))
