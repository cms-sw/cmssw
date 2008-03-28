# The following comments couldn't be translated into the new config version:

# Track reconstruction, using cosmictrackfinder

import FWCore.ParameterSet.Config as cms

# tracker geometry
#  include "Geometry/TrackerRecoData/data/trackerRecoGeometryXML.cfi"
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
#  include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi" 
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
#Tracking
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# COMBINATORIAL TRACK FINDER FOR COSMICS
#seeding
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmics_cfi import *
#/other options: CkfTIBD+ (set to CkfTIBD+ to run on slice test data or CkfTOB for TOB slice test)
#replace combinatorialcosmicseedfinder.GeometricStructure = CkfTIBD+  ## << NO, parameter no longer exists since 15x (gpetrucc)
# ^^ see http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/RecoTracker/SpecialSeedGenerators/data/CombinatorialSeedGeneratorForCosmics.cfi?r1=1.6&r2=1.3
#track candidates
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
#final fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
#track info
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
# COSMIC TRACK FINDER
# Seed generation
cosmicseedfinder = cms.EDFilter("CosmicSeedGenerator",
    originHalfLength = cms.double(90.0),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('TIBD+'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    TTRHBuilder = cms.string('WithTrackAngle'),
    ptMin = cms.double(0.9),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    originRadius = cms.double(150.0),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit")
)

cosmictrackfinder = cms.EDFilter("CosmicTrackFinder",
    TrajInEvents = cms.bool(True),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    HitProducer = cms.string('siStripRecHits'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MinHits = cms.int32(3),
    Chi2Cut = cms.double(100.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    GeometricStructure = cms.untracked.string('MTCC'),
    cosmicSeeds = cms.InputTag("cosmicseedfinder")
)

cluster_to_cosmictrack = cms.Sequence(cosmicseedfinder*cosmictrackfinder*trackinfo)
cluster_to_ckftrack = cms.Sequence(combinatorialcosmicseedfinder*ckfTrackCandidates*ctfWithMaterialTracks*trackinfo)
#replace MeasurementTrackerESProducer.pixelClusterProducer = ""   ### << NO, this is not the module name
MeasurementTracker.pixelClusterProducer = ''
ckfTrackCandidates.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilder'
ckfTrackCandidates.SeedProducer = 'combinatorialcosmicseedfinder'
ckfBaseTrajectoryFilter.filterPset.minPt = 0.01
ckfBaseTrajectoryFilter.filterPset.maxLostHits = 3
ckfBaseTrajectoryFilter.filterPset.maxConsecLostHits = 1
ckfBaseTrajectoryFilter.filterPset.minimumNumberOfHits = 3
ctfWithMaterialTracks.TrajectoryInEvent = True

