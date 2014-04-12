import FWCore.ParameterSet.Config as cms

def customiseForRunI(process):

    # apply only in reco step
    if not hasattr(process,'reconstruction'):
        return process

    # Put back 2012 default tracking. This piece of code is ugly.

    # first remove the current/default version of trackingGlocalReco
    # and delete all its descendent sequences that are going to be
    # redefined later on by the new process.load()

    # apply only in reco step
    if not hasattr(process,'reconstruction'):
        return process

    tgrIndex = process.globalreco.index(process.trackingGlobalReco)
    tgrIndexFromReco = process.reconstruction_fromRECO.index(process.trackingGlobalReco)
    process.globalreco.remove(process.trackingGlobalReco)
    process.reconstruction_fromRECO.remove(process.trackingGlobalReco)
    del process.trackingGlobalReco
    del process.ckftracks
    del process.ckftracks_wodEdX
    del process.ckftracks_plus_pixelless
    del process.ckftracks_woBH
    del process.iterTracking
    del process.InitialStep
    del process.LowPtTripletStep
    del process.PixelPairStep
    del process.DetachedTripletStep
    del process.MixedTripletStep
    del process.PixelLessStep
    del process.TobTecStep

    # Load the new Iterative Tracking configuration
    process.load("RecoTracker.Configuration.RecoTrackerRunI_cff")

    process.globalreco.insert(tgrIndex, process.trackingGlobalReco)
    process.reconstruction_fromRECO.insert(tgrIndexFromReco, process.trackingGlobalReco)

    return process
