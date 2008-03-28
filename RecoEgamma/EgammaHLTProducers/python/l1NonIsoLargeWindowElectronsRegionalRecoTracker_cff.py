import FWCore.ParameterSet.Config as cms

import copy
from RecoEgamma.EgammaHLTProducers.hltEgammaRegionalPixelSeedGenerator_cfi import *
# regional seeding 
# act locally, not globally!  so regional pixel tracking, let's go:
l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator = copy.deepcopy(hltEgammaRegionalPixelSeedGenerator)
# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cff import *
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate Ckf track candidates ############
l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates = copy.deepcopy(ckfTrackCandidates)
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
l1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = copy.deepcopy(hltEgammaRegionalCTFFinalFitWithMaterial)
#-------------------------------------------
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
from RecoTracker.RoadSearchCloudCleaner.CleanRoadSearchClouds_cff import *
# RoadSearchTrackCandidateMaker
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
# RS track fit with material 
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *
l1NonIsoLargeWindowElectronsRegionalRecoTracker = cms.Sequence(l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator*l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates*l1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)
l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator.candTag = 'l1NonIsoRecoEcalCandidate'
l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator.candTagEle = 'pixelMatchElectronsL1NonIsoLargeWindowForHLT'
l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator.UseZInVertex = True
l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator.originHalfLength = 0.5
#bool   seedCleaning         = false
l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates.SeedProducer = 'l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator'
l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates.SeedLabel = ''
l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedHits'
# set it as "none" to avoid redundant seed cleaner
l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner = 'CachingSeedCleanerBySharedInput'
#replace l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates.RedundantSeedCleaner  = "none"
l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates.NavigationSchool = 'SimpleNavigationSchool'
l1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial.src = 'l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates'

