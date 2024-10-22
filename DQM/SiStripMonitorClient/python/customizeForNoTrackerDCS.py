"""
Module to remove SiStrip DCS checks in Strip and Tracking Monitors
"""

import FWCore.ParameterSet.Config as cms

def producers_by_type(process, *types):
    return [module for module in process._Process__producers.values() if module._TypedParameterizable__type in types]

def removeDCSChecks(process, acceptedParts):
    print('WARNING: removing SiStrip DCS Checks in Strip and Tracking Monitors')

    for producerType in ['SiStripMonitorTrack', 'SiStripMonitorCluster']:
        for producer in producers_by_type(process, producerType):
            producer.UseDCSFiltering = cms.bool(False)
                    
    for producer in producers_by_type(process, 'SiStripMonitorCluster'):
        producer.StripDCSfilter.dcsPartitions = cms.vint32(acceptedParts)

    for producer in producers_by_type(process, 'TrackingMonitor'):
        producer.genericTriggerEventPSet.dcsPartitions = cms.vint32(acceptedParts)

    return process

def removeStripDCSChecks(process):
    removeDCSChecks(process, [28, 29])  # keep 28-29: pixel
    return process

def removePixelDCSChecks(process):
    removeDCSChecks(process, [24, 25, 26, 27])  # keep 24-27: strip
    return process

def removeTrackerDCSChecks(process):
    removeDCSChecks(process, []) # do not keep anything
    return process
