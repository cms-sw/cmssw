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
#  include "RecoTracker/SpecialSeedGenerators/data/CombinatorialSeedGeneratorForCosmicsTIFTIB.cff"
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsTIFTOB_cff import *
#/other options: CkfTIBD+ (set to CkfTIBD+ to run on slice test data or CkfTOB for TOB slice test)
#  replace combinatorialcosmicseedfinder.GeometricStructure = "CkfTOB"
#track candidates
from RecoTracker.CkfPattern.CkfTrackCandidatesTIFTOB_cff import *
import copy
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
trackinfoCTFTIFTOB = copy.deepcopy(trackinfo)
#    sequence cluster_to_cosmictrack = { cosmicseedfinder, cosmictrackfinder ,trackinfo }
cluster_to_ckftrack = cms.Sequence(cms.SequencePlaceholder("combinatorialcosmicseedfinder")*cms.SequencePlaceholder("ckfTrackCandidates")*cms.SequencePlaceholder("ctfWithMaterialTracksTIFTOB")*trackinfoCTFTIFTOB)
# We do this by hand because the cff requires all three algorithms to be defined
#    include "AnalysisAlgos/TrackInfoProducer/data/TrackInfoProducerTIFTOB.cff"
ctfWithMaterialTracksTIFTOB.TrajectoryInEvent = True
trackinfoCTFTIFTOB.cosmicTracks = 'ctfWithMaterialTracksTIFTOB'
trackinfoCTFTIFTOB.rechits = 'ctfWithMaterialTracksTIFTOB'

