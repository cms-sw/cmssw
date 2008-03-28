import FWCore.ParameterSet.Config as cms

# $Id: pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT.cff,v 1.3 2008/03/22 12:32:51 dkcira Exp $
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
l1NonIsoLargeWindowElectronPixelSeeds = copy.deepcopy(electronPixelSeeds)
from RecoEgamma.EgammaHLTProducers.pixelSeedConfigurationsForHLT_cfi import *
from RecoEgamma.EgammaHLTProducers.pixelMatchElectronsL1NonIsoLargeWindowForHLT_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# CKFTrackCandidateMaker
ckfL1NonIsoLargeWindowTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# CTF track fit with material
ctfL1NonIsoLargeWindowWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT = cms.Sequence(l1NonIsoLargeWindowElectronPixelSeeds)
pixelMatchElectronL1NonIsoLargeWindowTrackingSequenceForHLT = cms.Sequence(ckfL1NonIsoLargeWindowTrackCandidates+ctfL1NonIsoLargeWindowWithMaterialTracks+pixelMatchElectronsL1NonIsoLargeWindowForHLT)
l1NonIsoLargeWindowElectronPixelSeeds.SeedConfiguration = cms.PSet(
    l1NonIsoLargeWindowElectronPixelSeedConfiguration
)
l1NonIsoLargeWindowElectronPixelSeeds.barrelSuperClusters = 'correctedHybridSuperClustersL1NonIsolated'
l1NonIsoLargeWindowElectronPixelSeeds.endcapSuperClusters = 'correctedEndcapSuperClustersWithPreshowerL1NonIsolated'
ckfL1NonIsoLargeWindowTrackCandidates.SeedProducer = 'l1NonIsoLargeWindowElectronPixelSeeds'
ctfL1NonIsoLargeWindowWithMaterialTracks.src = 'ckfL1NonIsoLargeWindowTrackCandidates'

