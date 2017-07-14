# This file provides additional helpers for getting information of
# iterations in automated way.
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
import RecoTracker.IterativeTracking.iterativeTk_cff as _iterativeTk_cff

def getMVASelectors(postfix):
    # assume naming convention that the iteration name (when first
    # letter in lower case) is the selector name

    ret = {}

    for iterName, seqName in _cfg.iterationAlgos(postfix, includeSequenceName=True):
        if hasattr(_iterativeTk_cff, iterName):
            mod = getattr(_iterativeTk_cff, iterName)
            seq = getattr(_iterativeTk_cff, seqName)

            # Ignore iteration if the MVA selector module is not in the sequence
            try:
                seq.index(mod)
            except:
                continue

            typeName = mod._TypedParameterizable__type
            classifiers = []
            if typeName == "ClassifierMerger":
                classifiers = mod.inputClassifiers.value()
            elif "TrackMVAClassifier" in typeName:
                classifiers = [iterName]
            if len(classifiers) > 0:
                ret[iterName] = (iterName+"Tracks", classifiers)

    return ret
