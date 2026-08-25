import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import esproducers_by_type

def customizeNoBPixShapeCutL1Ends(process):
    regions = cms.VPSet(
        cms.PSet(modules = cms.vuint32(1, 2, 7, 8), # ladders are omitted == all ladders
                 layers  = cms.vuint32(1)),
    )
    for prod in esproducers_by_type(process, 'ClusterShapeHitFilterESProducer'):
        prod.noBPixShapeCutRegions = regions.copy()
    return process

def customizeBPixShapeCutL1CenterOnly(process):
    regions = cms.VPSet(
        cms.PSet(modules = cms.vuint32(1, 2, 3, 6, 7, 8), # ladders are omitted == all ladders
                 layers  = cms.vuint32(1)),
    )
    for prod in esproducers_by_type(process, 'ClusterShapeHitFilterESProducer'):
        prod.noBPixShapeCutRegions = regions.copy()
    return process

def customizeNoBPixShapeCutL1(process):
    regions = cms.VPSet(
        cms.PSet(layers  = cms.vuint32(1))
    )
    for prod in esproducers_by_type(process, 'ClusterShapeHitFilterESProducer'):
        prod.noBPixShapeCutRegions = regions.copy()
    return process