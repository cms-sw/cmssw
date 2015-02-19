import FWCore.ParameterSet.Config as cms

#
# Define Validation sequence over "highPurity" tracks. 
# For proper comparison the same requirement should be applied on FullSim as well.
#
# Tracking particle module
from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
# Track Associators
from SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociatorPRoducers.trackAssociatorByHits_cfi import *
#new postreco sequence
from SimTracker.TrackAssociation.trackMCMatch_cfi import *
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff import *
# Track Validator    
from Validation.RecoTrack.cuts_cff import *
from Validation.RecoTrack.cutsTPEffic_cfi import *
from Validation.RecoTrack.cutsTPFake_cfi import *
from Validation.RecoTrack.MultiTrackValidator_cff import *
valid = cms.Sequence(cms.SequencePlaceholder("genParticles")*trackingParticles*cutsRecoTracks*cutsTPEffic*cutsTPFake*trackAssociatorByHits*multiTrackValidator)
#mergedtruth.simHitCollections = cms.PSet(tracker = cms.vstring("famosSimHitsTrackerHits"))
#mergedtruth.simHitLabel = 'famosSimHits'
mergedtruth.removeDeadModules = cms.bool(False)
trackAssociatorByHits.associateStrip = False
trackAssociatorByHits.associatePixel = False
#TrackAssociatorByHits.ROUList = ['famosSimHitsTrackerHits']

#use cutsRecoTracks
cutsRecoTracks.quality = ['highPurity']

# pass only high purity tracks
multiTrackValidator.label = ['cutsRecoTracks']
multiTrackValidator.associators = ['trackAssociatorByHits']
multiTrackValidator.UseAssociators = True


