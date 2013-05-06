import FWCore.ParameterSet.Config as cms

def customise_trackMon_IterativeTracking_2012(process):

    if hasattr(process,"trackMonIterativeTracking2012"):
        print "trackMonIterativeTracking2012 DEFINED !!!"
    else :
        print "trackMonIterativeTracking2012 NOT DEFINED !!!"
    
    from DQM.TrackingMonitor.TrackingMonitorSeedNumber_cff import trackMonIterativeTracking2012

    if hasattr(process,"SiStripDQMTier0") and hasattr(process,"trackMonIterativeTracking2012"):
        getattr(process,"SiStripDQMTier0").replace(
            getattr(process,"trackMonIterativeTracking2012"),
            trackMonIterativeTracking2012
        )

    if hasattr(process,"SiStripDQMTier0Common") and hasattr(process,"trackMonIterativeTracking2012"):
        getattr(process,"SiStripDQMTier0Common").replace(
            getattr(process,"trackMonIterativeTracking2012"),
            trackMonIterativeTracking2012
        )

    if hasattr(process,"SiStripDQMTier0MinBias") and hasattr(process,"trackMonIterativeTracking2012"):
        getattr(process,"SiStripDQMTier0MinBias").replace(
            getattr(process,"trackMonIterativeTracking2012"),
            trackMonIterativeTracking2012
            )

    return process

def customise_trackMon_IterativeTracking_PHASE1(process):

    if hasattr(process,"trackMonIterativeTracking2012"):
        print "trackMonIterativeTracking2012 DEFINED !!!"
    else :
        print "trackMonIterativeTracking2012 NOT DEFINED !!!"
    
#    from DQM.TrackingMonitor.TrackingMonitorSeedNumber_PhaseI_cff import * # trackMonIterativeTrackingPhaseI

    process.load("DQM.TrackingMonitor.TrackingMonitorSeedNumber_PhaseI_cff")

    if hasattr(process,"SiStripDQMTier0") and hasattr(process,"trackMonIterativeTracking2012"):
        getattr(process,"SiStripDQMTier0").replace(
            getattr(process,"trackMonIterativeTracking2012"),
            getattr(process,"trackMonIterativeTrackingPhaseI")
        )

    if hasattr(process,"SiStripDQMTier0Common") and hasattr(process,"trackMonIterativeTracking2012"):
        getattr(process,"SiStripDQMTier0Common").replace(
            getattr(process,"trackMonIterativeTracking2012"),
            getattr(process,"trackMonIterativeTrackingPhaseI")
        )

    if hasattr(process,"SiStripDQMTier0MinBias") and hasattr(process,"trackMonIterativeTracking2012"):
        getattr(process,"SiStripDQMTier0MinBias").replace(
            getattr(process,"trackMonIterativeTracking2012"),
            getattr(process,"trackMonIterativeTrackingPhaseI")
            )

    return process
