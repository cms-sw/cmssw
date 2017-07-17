import FWCore.ParameterSet.Config as cms

#### Configuration of the track handling modules as in Alignment/OfflineValidation

from RecoTracker.TrackProducer.TrackRefitters_cff import *
refittedTracks = TrackRefitter.clone(src = cms.InputTag("generalTracks"))

# try to reproduce Alignment/OfflineValidation

from RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff import *
TrackerTrackHitFilter.src = 'refittedTracks'
TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
TrackerTrackHitFilter.minimumHits = 8
TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
TrackerTrackHitFilter.detsToIgnore = []
TrackerTrackHitFilter.replaceWithInactiveHits = True
TrackerTrackHitFilter.stripAllInvalidHits = False
TrackerTrackHitFilter.rejectBadStoNHits = True
TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 14.0")
TrackerTrackHitFilter.rejectLowAngleHits= True
TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
TrackerTrackHitFilter.usePixelQualityFlag= True

################################################################################################
#TRACK PRODUCER
#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
################################################################################################
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff   #TrackRefitters_cff
HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
    src = 'TrackerTrackHitFilter',
    #TrajectoryInEvent = True
    TTRHBuilder = "WithAngleAndTemplate"
)

from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
AlignmentTrackSelector.src ='HitFilteredTracks' 
AlignmentTrackSelector.applyBasicCuts = True
AlignmentTrackSelector.ptMin   = 1.5
AlignmentTrackSelector.pMin   = 0.
AlignmentTrackSelector.nHitMin =10
AlignmentTrackSelector.nHitMin2D = 2
AlignmentTrackSelector.chi2nMax = 100.

refittedATSTracks = TrackRefitter.clone(src = cms.InputTag("AlignmentTrackSelector"))

seqTrackRefitting = cms.Sequence( refittedTracks
                                  + TrackerTrackHitFilter
                                  + HitFilteredTracks
                                  + AlignmentTrackSelector
                                  + refittedATSTracks
                                  )
