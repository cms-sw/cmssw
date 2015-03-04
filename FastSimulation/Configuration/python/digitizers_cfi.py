#######################
# this file is the FastSim equivalent of SimGeneral/MixingModule/python/digitizers_cfi.py
# author: Lukas Vanelderen
# date:   Jan 20 2015
#######################

import FWCore.ParameterSet.Config as cms

#######################
# CONFIGURE DIGITIZERS / TRACK ACCUMULATOR / TRUTH ACCUMULATOR
#######################

def digitizersFull2Fast(digitizers):
    # fastsim does not simulate castor
    if hasattr(digitizers,"castor"):
        delattr(digitizers,"castor")
    else:
        print "WARNING: digitizers has no attribute 'castor'"
        
    # fastsim does not digitize pixel and strip hits, it mixes tracks
    if hasattr(digitizers,"pixel") and hasattr(digitizers,"strip"):
        delattr(digitizers,"pixel")
        delattr(digitizers,"strip")
        import FastSimulation.Tracking.recoTrackAccumulator_cfi
        digitizers.tracker = cms.PSet(FastSimulation.Tracking.recoTrackAccumulator_cfi.recoTrackAccumulator)
    else:
        print "WARNING: digitizers has no attribute 'pixel' and/or 'strip'"
        print "       : => not mixing tracks"

    # fastsim has its own names for simhit collections
    for element in ["ecal","hcal"]:
        if hasattr(digitizers,element):
            getattr(digitizers,element).hitsProducer = "famosSimHits"
        else:
            print "WARNING: digitizers has no attribute '{0}'".format(element)
            
    # fastsim has different input for merged truth
    if hasattr(digitizers,"mergedtruth"):
        digitizers.mergedtruth.allowDifferentSimHitProcesses = True
        digitizers.mergedtruth.simHitCollections = cms.PSet(
            muon = cms.VInputTag( cms.InputTag('MuonSimHits','MuonDTHits'),
                                  cms.InputTag('MuonSimHits','MuonCSCHits'),
                                  cms.InputTag('MuonSimHits','MuonRPCHits') ),
            trackerAndPixel = cms.VInputTag( cms.InputTag('famosSimHits','TrackerHits') )
            )
        digitizers.mergedtruth.simTrackCollection = cms.InputTag('famosSimHits')
        digitizers.mergedtruth.simVertexCollection = cms.InputTag('famosSimHits')

    return digitizers

import SimGeneral.MixingModule.digitizers_cfi

theDigitizersValid = digitizersFull2Fast(SimGeneral.MixingModule.digitizers_cfi.theDigitizersValid)
theDigitizers = digitizersFull2Fast(SimGeneral.MixingModule.digitizers_cfi.theDigitizers)

#######################
# ALIASES FOR DIGI AND MIXED TRACK COLLECTIONS
#######################

# simEcalUnsuppressedDigis:   alias for ECAL digis produced by MixingModule
# simHcalUnsuppressedDigis:   alias for HCAL digis produced by MixingModule

from SimGeneral.MixingModule.digitizers_cfi import simEcalUnsuppressedDigis,simHcalUnsuppressedDigis
# alias for collections of tracks , track extras and tracker hits produced by MixingModule 
generalTracks = cms.EDAlias(
    mix = cms.VPSet( cms.PSet(type=cms.string('recoTracks'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ),
                     cms.PSet(type=cms.string('recoTrackExtras'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ),
                     cms.PSet(type=cms.string('TrackingRecHitsOwned'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ) )
    )


