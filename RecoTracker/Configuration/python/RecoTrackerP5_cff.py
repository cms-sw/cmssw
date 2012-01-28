import FWCore.ParameterSet.Config as cms

#
# Tracking configuration file fragment for P5 cosmic running
#
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import *
# TTRHBuilders
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
# COSMIC TRACK FINDER
from RecoTracker.SpecialSeedGenerators.CosmicSeedP5Pairs_cff import *
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import *
# Final Track Selector for CosmicTF
from RecoTracker.FinalTrackSelectors.CosmicTFFinalTrackSelectorP5_cff import *

#chi2 set to 40!!
# CTF
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsP5_cff import *
from RecoTracker.SpecialSeedGenerators.SimpleCosmicBONSeeder_cfi import *
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
combinedP5SeedsForCTF = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
combinedP5SeedsForCTF.seedCollections = cms.VInputTag(
    cms.InputTag('combinatorialcosmicseedfinderP5'),
    cms.InputTag('simpleCosmicBONSeeds'),
)
#backward compatibility 2.2/3.1
combinedP5SeedsForCTF.PairCollection = cms.InputTag('combinatorialcosmicseedfinderP5')
combinedP5SeedsForCTF.TripletCollection = cms.InputTag('simpleCosmicBONSeeds')

from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import *
ckfTrackCandidatesP5.src = cms.InputTag('combinedP5SeedsForCTF')
#backward compatibility 2.2/3.1
#ckfTrackCandidatesP5.SeedProducer = 'combinedP5SeedsForCTF'

#import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import *
# Final Track Selector for CTF
from RecoTracker.FinalTrackSelectors.CTFFinalTrackSelectorP5_cff import *

# ROACH SEARCH
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeedsP5_cff import *
from RecoTracker.RoadSearchCloudMaker.RoadSearchCloudsP5_cff import *
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidatesP5_cff import *
from RecoTracker.TrackProducer.RSFinalFitWithMaterialP5_cff import *
# Final Track Selector for RS
from RecoTracker.FinalTrackSelectors.RSFinalTrackSelectorP5_cff import *

# TRACK INFO
#include "AnalysisAlgos/TrackInfoProducer/data/TrackInfoProducerP5.cff"

ckfTrackCandidatesP5LHCNavigation    = ckfTrackCandidatesP5.clone(NavigationSchool = cms.string('SimpleNavigationSchool'))
ctfWithMaterialTracksP5LHCNavigation = ctfWithMaterialTracksCosmics.clone(src = cms.InputTag("ckfTrackCandidatesP5LHCNavigation"))

ctftracksP5 = cms.Sequence(combinatorialcosmicseedfinderP5*simpleCosmicBONSeeds*combinedP5SeedsForCTF*
                           ckfTrackCandidatesP5*ctfWithMaterialTracksCosmics*ctfWithMaterialTracksP5+
                           ckfTrackCandidatesP5LHCNavigation*ctfWithMaterialTracksP5LHCNavigation)

rstracksP5 = cms.Sequence(roadSearchSeedsP5*roadSearchCloudsP5*rsTrackCandidatesP5*rsWithMaterialTracksCosmics*rsWithMaterialTracksP5)

from RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi import *
cosmicTrackSplitter.tjTkAssociationMapTag = 'cosmictrackfinderCosmics'
cosmicTrackSplitter.tracks = 'cosmictrackfinderCosmics'
splittedTracksP5 = cosmictrackfinderCosmics.clone(src = cms.InputTag("cosmicTrackSplitter"))
    
cosmictracksP5 = cms.Sequence(cosmicseedfinderP5*cosmicCandidateFinderP5*cosmictrackfinderCosmics*cosmictrackfinderP5*cosmicTrackSplitter*splittedTracksP5)


#Top/Bottom tracks NEW
from RecoTracker.Configuration.RecoTrackerTopBottom_cff import *
trackerCosmics_TopBot = cms.Sequence((trackerlocalrecoTop*tracksP5Top)+(trackerlocalrecoBottom*tracksP5Bottom))

#dEdX reconstruction
from RecoTracker.DeDx.dedxEstimators_Cosmics_cff import *
#sequence tracksP5 = {cosmictracksP5, ctftracksP5, rstracksP5, trackinfoP5}
# (SK) keep rstracks commented out in case of resurrection
#tracksP5 = cms.Sequence(cosmictracksP5*ctftracksP5*rstracksP5*trackerCosmics_TopBot*doAllCosmicdEdXEstimators)
tracksP5 = cms.Sequence(cosmictracksP5*ctftracksP5*trackerCosmics_TopBot*doAllCosmicdEdXEstimators)
tracksP5_wodEdX = tracksP5.copy()
tracksP5_wodEdX.remove(doAllCosmicdEdXEstimators)

# explicitely switch on hit splitting
ckfTrackCandidatesP5.useHitsSplitting = True
rsTrackCandidatesP5.SplitMatchedHits = True



# REGIONAL RECONSTRUCTION
from RecoTracker.Configuration.RecoTrackerNotStandard_cff import regionalCosmicTrackerSeeds,regionalCosmicCkfTrackCandidates,regionalCosmicTracks,regionalCosmicTracksSeq
regionalCosmicTrackerSeeds.RegionInJetsCheckPSet = cms.PSet( doJetsExclusionCheck   = cms.bool( False ) )
