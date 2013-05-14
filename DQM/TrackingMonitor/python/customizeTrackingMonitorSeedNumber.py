import FWCore.ParameterSet.Config as cms

def customise_trackMon_IterativeTracking_2012(process):

## DEBUGGING
#    if hasattr(process,"trackMonIterativeTracking2012"):
#        print "trackMonIterativeTracking2012 DEFINED !!!"
#    else :
#        print "trackMonIterativeTracking2012 NOT DEFINED !!!"
#    print "IterativeTracking_2012"    

    from DQM.TrackingMonitor.TrackingMonitorSeedNumber_cff import trackMonIterativeTracking2012

    for s in ["SiStripDQMTier0", "SiStripDQMTier0Common", "SiStripDQMTier0MinBias"] :
        idx = getattr(process,s).index(getattr(process,"TrackMonStep0"))
        getattr(process,s).remove(getattr(process,"TrackMonStep0")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep1")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep2")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep3")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep4")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep5")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep6")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep9")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep10") )
           
        getattr(process,s).insert(idx,getattr(process,"trackMonIterativeTracking2012"))

    return process

######### Phase1
def customise_trackMon_IterativeTracking_PHASE1(process):

## DEBUGGING
#    if hasattr(process,"trackMonIterativeTracking2012"):
#        print "trackMonIterativeTracking2012 DEFINED !!!"
#    else :
#        print "trackMonIterativeTracking2012 NOT DEFINED !!!"
#    print "IterativeTracking_PHASE1"    

    process.load("DQM.TrackingMonitor.TrackingMonitorSeedNumber_PhaseI_cff")

    for s in ["SiStripDQMTier0", "SiStripDQMTier0Common", "SiStripDQMTier0MinBias"] :
        idx = getattr(process,s).index(getattr(process,"TrackMonStep0"))
        getattr(process,s).remove(getattr(process,"TrackMonStep0")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep1")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep2")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep3")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep4")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep5")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep6")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep9")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep10") )
           
        getattr(process,s).insert(idx,getattr(process,"trackMonIterativeTrackingPhaseI"))

    return process
        
######## Phase1 PU70
def customise_trackMon_IterativeTracking_PHASE1PU70(process):

## DEBUGGING
#    if hasattr(process,"trackMonIterativeTracking2012"):
#        print "trackMonIterativeTracking2012 DEFINED !!!"
#    else :
#        print "trackMonIterativeTracking2012 NOT DEFINED !!!"
#    print "IterativeTracking_PHASE1_PU70"    

    process.load("DQM.TrackingMonitor.TrackingMonitorSeedNumber_Phase1PU70_cff")

    for s in ["SiStripDQMTier0", "SiStripDQMTier0Common", "SiStripDQMTier0MinBias"] :
        idx = getattr(process,s).index(getattr(process,"TrackMonStep0"))
        getattr(process,s).remove(getattr(process,"TrackMonStep0")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep1")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep2")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep3")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep4")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep5")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep6")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep9")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep10") )
           
        getattr(process,s).insert(idx,getattr(process,"trackMonIterativeTrackingPhase1PU70"))
            
    return process

######## Phase1 PU140
def customise_trackMon_IterativeTracking_PHASE1PU140(process):

## DEBUGGING
#    if hasattr(process,"trackMonIterativeTracking2012"):
#        print "trackMonIterativeTracking2012 DEFINED !!!"
#    else :
#        print "trackMonIterativeTracking2012 NOT DEFINED !!!"
#    print "IterativeTracking_PHASE1_PU140"    

    process.load("DQM.TrackingMonitor.TrackingMonitorSeedNumber_Phase1PU140_cff")

    for s in ["SiStripDQMTier0", "SiStripDQMTier0Common", "SiStripDQMTier0MinBias"] :
        idx = getattr(process,s).index(getattr(process,"TrackMonStep0"))
        getattr(process,s).remove(getattr(process,"TrackMonStep0")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep1")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep2")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep3")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep4")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep5")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep6")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep9")  )
        getattr(process,s).remove(getattr(process,"TrackMonStep10") )
           
        getattr(process,s).insert(idx,getattr(process,"trackMonIterativeTrackingPhase1PU140"))

    return process
