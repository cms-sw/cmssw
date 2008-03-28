import FWCore.ParameterSet.Config as cms

import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaRegionalPixelSeedGenerator_cfi import *
# regional seeding 
# act locally, not globally!  so regional pixel tracking, let's go:
l1IsoLargeWindowElectronsRegionalPixelSeedGenerator = copy.deepcopy(hltEgammaRegionalPixelSeedGenerator)
# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cff import *
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate Ckf track candidates ############
l1IsoLargeWindowElectronsRegionalCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
#-------------------------------------------------------------------	
# generate CTF track candidates
# magnetic field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# cms geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaRegionalCTFFinalFitWithMaterial_cfi import *
# TrackProducer
l1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = copy.deepcopy(hltEgammaRegionalCTFFinalFitWithMaterial)
#-------------------------------------------
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
from RecoTracker.RoadSearchCloudCleaner.CleanRoadSearchClouds_cff import *
# RoadSearchTrackCandidateMaker
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
# RS track fit with material 
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *
l1IsoLargeWindowElectronsRegionalRecoTracker = cms.Sequence(l1IsoLargeWindowElectronsRegionalPixelSeedGenerator*l1IsoLargeWindowElectronsRegionalCkfTrackCandidates*l1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)
l1IsoLargeWindowElectronsRegionalPixelSeedGenerator.candTag = 'l1IsoRecoEcalCandidate'
l1IsoLargeWindowElectronsRegionalPixelSeedGenerator.candTagEle = 'pixelMatchElectronsL1IsoLargeWindowForHLT'
l1IsoLargeWindowElectronsRegionalPixelSeedGenerator.UseZInVertex = True
l1IsoLargeWindowElectronsRegionalPixelSeedGenerator.originHalfLength = 0.5
#bool   seedCleaning         = false
l1IsoLargeWindowElectronsRegionalCkfTrackCandidates.SeedProducer = 'l1IsoLargeWindowElectronsRegionalPixelSeedGenerator'
l1IsoLargeWindowElectronsRegionalCkfTrackCandidates.SeedLabel = ''
l1IsoLargeWindowElectronsRegionalCkfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
l1IsoLargeWindowElectronsRegionalCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
# set it as "none" to avoid redundant seed cleaner
l1IsoLargeWindowElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
#replace l1IsoLargeWindowElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner  = "none"
l1IsoLargeWindowElectronsRegionalCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'
l1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial.src = 'l1IsoLargeWindowElectronsRegionalCkfTrackCandidates'

