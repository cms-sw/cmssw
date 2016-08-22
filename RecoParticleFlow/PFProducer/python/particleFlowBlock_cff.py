import FWCore.ParameterSet.Config as cms

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
# include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
from RecoParticleFlow.PFProducer.particleFlowBlock_cfi import *

from Configuration.StandardSequences.Eras import eras
_phase2_hgcal_Importers = particleFlowBlock.elementImporters.copy()
_phase2_hgcal_Importers[2].source_ee = cms.InputTag('particleFlowSuperClusterHGCal')
_phase2_hgcal_Importers.append(
    cms.PSet( importerName = cms.string("HGCalClusterImporter"),
              source = cms.InputTag("particleFlowClusterHGCal"),
              BCtoPFCMap = cms.InputTag('particleFlowSuperClusterHGCal:PFClusterAssociationEBEE') ),
)
_phase2_hgcal_Linkers = particleFlowBlock.linkDefinitions.copy()
_phase2_hgcal_Linkers.append( 
    cms.PSet( linkerName = cms.string("SCAndHGCalLinker"),
              linkType   = cms.string("SC:HGCAL"),
              useKDTree  = cms.bool(False),
              SuperClusterMatchByRef = cms.bool(True) ) 
)
_phase2_hgcal_Linkers.append(
    cms.PSet( linkerName = cms.string("HGCalAndBREMLinker"),
              linkType   = cms.string("HGCAL:BREM"),
              useKDTree  = cms.bool(False) )
)
_phase2_hgcal_Linkers.append(
    cms.PSet( linkerName = cms.string("GSFAndHGCalLinker"), 
                  linkType   = cms.string("GSF:HGCAL"),
                  useKDTree  = cms.bool(False) )
)

eras.phase2_hgcal.toModify(
    particleFlowBlock,
    linkDefinitions = _phase2_hgcal_Linkers,
    elementImporters = _phase2_hgcal_Importers
)
