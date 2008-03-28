import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchElectronL1IsoLargeWindowSequenceForHLT.cff,v 1.4 2008/03/22 12:32:50 dkcira Exp $
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# NB: it assumes that ECAL clusters (hybrid) are in the event
#
#
# initialize magnetic field #########################
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# initialize geometry #####################
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# ESProducers needed for tracking
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cff import *
# modules to make seeds, tracks and electrons
from RecoEgamma.EgammaHLTProducers.egammaHLTChi2MeasurementEstimatorESProducer_cff import *
from RecoEgamma.EgammaElectronProducers.electronPixelSeeds_cfi import *
import copy
from RecoEgamma.EgammaElectronProducers.electronPixelSeeds_cfi import *
l1IsoLargeWindowElectronPixelSeeds = copy.deepcopy(electronPixelSeeds)
from RecoEgamma.EgammaHLTProducers.pixelSeedConfigurationsForHLT_cfi import *
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronsL1IsoLargeWindowForHLT_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# CKFTrackCandidateMaker
ckfL1IsoLargeWindowTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# CTF track fit with material
ctfL1IsoLargeWindowWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
pixelMatchElectronL1IsoLargeWindowSequenceForHLT = cms.Sequence(l1IsoLargeWindowElectronPixelSeeds)
pixelMatchElectronL1IsoLargeWindowTrackingSequenceForHLT = cms.Sequence(ckfL1IsoLargeWindowTrackCandidates+ctfL1IsoLargeWindowWithMaterialTracks+pixelMatchElectronsL1IsoLargeWindowForHLT)
l1IsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
    l1IsoLargeWindowElectronPixelSeedConfiguration
)
l1IsoLargeWindowElectronPixelSeeds.barrelSuperClusters = 'correctedHybridSuperClustersL1Isolated'
l1IsoLargeWindowElectronPixelSeeds.endcapSuperClusters = 'correctedEndcapSuperClustersWithPreshowerL1Isolated'
ckfL1IsoLargeWindowTrackCandidates.SeedProducer = 'l1IsoLargeWindowElectronPixelSeeds'
ctfL1IsoLargeWindowWithMaterialTracks.src = 'ckfL1IsoLargeWindowTrackCandidates'

