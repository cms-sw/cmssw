import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_cff import *
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff import *

anaTrack.doSimVertex = True  
anaTrack.doSimTrack = True 
anaTrack.fillSimTrack = True

anaTrack.simTrackPtMin = 0.49

pixelTrack.simTrackPtMin = 0.4

ppTrack.doSimVertex = True
ppTrack.doSimTrack = True
ppTrack.fillSimTrack = True

ppTrack.simTrackPtMin = 0.49
ppTrack.associatorMap = cms.InputTag('tpRecoAssocGeneralTracks')

tpRecoAssocGeneralTracks = trackingParticleRecoTrackAsssociation.clone()
tpRecoAssocGeneralTracks.label_tr = cms.InputTag("generalTracks")

tpRecoAssocHiGeneralTracks = trackingParticleRecoTrackAsssociation.clone()
tpRecoAssocHiGeneralTracks.label_tr = cms.InputTag("hiGeneralTracks")
quickTrackAssociatorByHits.ComponentName = cms.string('quickTrackAssociatorByHits')

quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')
# quickTrackAssociatorByHits.Cut_RecoToSim = cms.double(0.5) # put this back in for 50% hit matching
quickTrackAssociatorByHits.Quality_SimToReco = cms.double(0.0)

trackSequencesPbPb = cms.Sequence(quickTrackAssociatorByHits +
                                  tpRecoAssocHiGeneralTracks +
								  anaTrack)
								  
trackSequencesPP = cms.Sequence(quickTrackAssociatorByHits +
                                tpRecoAssocGeneralTracks +
								ppTrack)
							
