import FWCore.ParameterSet.Config as cms

from copy import deepcopy

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
# include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
from RecoParticleFlow.PFProducer.particleFlowBlock_cfi import *

_phase2_hgcal_Importers = deepcopy(particleFlowBlock.elementImporters)
# kill tracks in the HGCal
_phase2_hgcal_Importers[5].importerName = cms.string('GeneralTracksImporterWithVeto')
_phase2_hgcal_Importers[5].veto = cms.InputTag('hgcalTrackCollection:TracksInHGCal')
#_phase2_hgcal_Importers[2].source_ee = cms.InputTag('particleFlowSuperClusterHGCal')
#_phase2_hgcal_Importers.append(
#    cms.PSet( importerName = cms.string("HGCalClusterImporter"),
#              source = cms.InputTag("particleFlowClusterHGCal"),
#              BCtoPFCMap = cms.InputTag('particleFlowSuperClusterHGCal:PFClusterAssociationEBEE') ),
#)
_phase2_hgcal_Linkers = deepcopy(particleFlowBlock.linkDefinitions)
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

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(
    particleFlowBlock,
    linkDefinitions = _phase2_hgcal_Linkers,
    elementImporters = _phase2_hgcal_Importers
)
