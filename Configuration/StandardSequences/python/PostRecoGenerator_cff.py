import FWCore.ParameterSet.Config as cms

#track match
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.trackMCMatchSequence_cff import *
# define post-reco generator sequence
postreco_generator = cms.Sequence(trackMCMatchSequence)

