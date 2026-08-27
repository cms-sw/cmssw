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

# Cluster shape payloads retuned for the 2026 Run 3 pixel detector.
# The L1 file is only relevant when the shape cut is applied on BPix L1,
# i.e. it is loaded but unused on top of customizeNoBPixShapeCutL1.
_run3_2026_PixelShapeFile = 'RecoTracker/PixelLowPtUtilities/data/data/run3_2026_L2_L3_L4.par'
_run3_2026_PixelShapeFileL1 = 'RecoTracker/PixelLowPtUtilities/data/data/run3_2026_L1.par'

def customizeRun3_2026PixelShapePayloads(process):
    for prod in esproducers_by_type(process, 'ClusterShapeHitFilterESProducer'):
        prod.PixelShapeFile = _run3_2026_PixelShapeFile
        prod.PixelShapeFileL1 = _run3_2026_PixelShapeFileL1
    return process

def customizeNoBPixShapeCutL1Run3_2026(process):
    process = customizeNoBPixShapeCutL1(process)
    process = customizeRun3_2026PixelShapePayloads(process)
    return process
