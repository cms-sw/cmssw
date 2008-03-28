import FWCore.ParameterSet.Config as cms

#
# ctf tracks parameter-set entries for module
#
# SimpleTrackListMerger
#
# located in
#
# RecoTracker/FinalTrackSelectors
#
# 
# sequence dependency:
#
# - rsWithMaterialTracks: include "RecoTracker/TrackProducer/data/RSFinalFitWithMaterial.cff"
#
#
# service dependency:
#
# - geometry:          include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
# - tracker geometry:  include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
# - tracker numbering: include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
#
# function:
#
# cleans and merges ctf and rs Track lists and put new list back in Event
rsMCTrackListCleaner = cms.EDFilter("SimpleTrackListMerger",
    # minimum shared fraction to be called duplicate
    ShareFrac = cms.double(0.66),
    # minimum pT in GeV/c
    MinPT = cms.double(0.05),
    # minimum difference in rechit position in cm
    # negative Epsilon uses sharedInput for comparison
    Epsilon = cms.double(-0.001),
    # maximum chisq/dof
    MaxNormalizedChisq = cms.double(20.0),
    # minimum number of RecHits used in fit
    MinFound = cms.int32(8),
    # will only clean TrackProducer1 if TrackProducer2 doesn't exist
    TrackProducer2 = cms.string('IgnoreThisWarning'),
    # module laber of RS Tracks from KF with material propagator
    TrackProducer1 = cms.string('rsWithMaterialTracks')
)


