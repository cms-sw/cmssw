import FWCore.ParameterSet.Config as cms

from copy import deepcopy

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
# include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
from RecoParticleFlow.PFProducer.particleFlowBlock_cfi import *

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
# kill tracks in the HGCal
phase2_hgcal.toModify(
    particleFlowBlock,
    elementImporters = { 5 : dict(
        importerName = cms.string('GeneralTracksImporterWithVeto'),
        veto = cms.InputTag('hgcalTrackCollection:TracksInHGCal')
    ) }
)
### for later
#_phase2_hgcal_Linkers.append( 
#    cms.PSet( linkerName = cms.string("SCAndHGCalLinker"),
#              linkType   = cms.string("SC:HGCAL"),
#              useKDTree  = cms.bool(False),
#              SuperClusterMatchByRef = cms.bool(True) ) 
#)
#_phase2_hgcal_Linkers.append(
#    cms.PSet( linkerName = cms.string("HGCalAndBREMLinker"),
#              linkType   = cms.string("HGCAL:BREM"),
#              useKDTree  = cms.bool(False) )
#)
#_phase2_hgcal_Linkers.append(
#    cms.PSet( linkerName = cms.string("GSFAndHGCalLinker"), 
#                  linkType   = cms.string("GSF:HGCAL"),
#                  useKDTree  = cms.bool(False) )
#)


from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    particleFlowBlock,
    elementImporters = { 5  : dict( timeValueMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModel"),
                                timeErrorMap = cms.InputTag("trackTimeValueMapProducer:generalTracksConfigurableFlatResolutionModelResolution")) }
)
