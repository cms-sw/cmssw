#######################
# this file is the FastSim equivalent of SimGeneral/MixingModule/python/digitizers_cfi.py
# author: Lukas Vanelderen
# date:   Jan 20 2015
#######################

import FWCore.ParameterSet.Config as cms

#######################
# CONFIGURE DIGITIZERS / TRACK ACCUMULATOR / TRUTH ACCUMULATOR
#######################

# theDigitizers:              digitizer configuration for MixingModule, used for production
# theDigitizersValid:         digitizer configuration for MixingModule, used for validation
from SimGeneral.MixingModule.digitizers_cfi import theDigitizers,theDigitizersValid

# fastsim has no digitization of tracker
del theDigitizers.pixel
del theDigitizers.strip

# fastsim does not model castor
del theDigitizers.castor

# fastsim hits and fullsim hits have different names
theDigitizers.ecal.hitsProducer = cms.string("famosSimHits")
theDigitizers.hcal.hitsProducer = cms.string("famosSimHits")

# fastsim mixes tracks
from FastSimulation.Tracking.recoTrackAccumulator_cfi import recoTrackAccumulator
theDigitizers.tracker = cms.PSet(recoTrackAccumulator)

# fastsim has different input for merged truth
mergedtruth = theDigitizersValid.mergedtruth
mergedtruth.allowDifferentSimHitProcesses = True
mergedtruth.simHitCollections = cms.PSet(
        muon = cms.VInputTag( cms.InputTag('MuonSimHits','MuonDTHits'),
                       cms.InputTag('MuonSimHits','MuonCSCHits'),
                       cms.InputTag('MuonSimHits','MuonRPCHits') ),
        trackerAndPixel = cms.VInputTag( cms.InputTag('famosSimHits','TrackerHits') )
    )
mergedtruth.simTrackCollection = cms.InputTag('famosSimHits')
mergedtruth.simVertexCollection = cms.InputTag('famosSimHits')

theDigitizersValid = theDigitizers.clone()
theDigitizersValid.mergedtruth = cms.PSet(mergedtruth)

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


