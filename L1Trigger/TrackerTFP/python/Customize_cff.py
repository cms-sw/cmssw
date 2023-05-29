import FWCore.ParameterSet.Config as cms

def setupUseTMTT(process):
    process.TrackTriggerDataFormats.UseHybrid = False
    process.TrackTriggerSetup.TrackingParticle.MinPt = 3.0
    process.TrackTriggerSetup.Firmware.MaxdPhi = 0.01
    return process