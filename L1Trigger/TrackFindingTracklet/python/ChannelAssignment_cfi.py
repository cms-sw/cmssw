# defines PSet to assign tracklet tracks and stubs to output channel based on their Pt or seed type as well as DTC stubs to input channel
import FWCore.ParameterSet.Config as cms

ChannelAssignment_params = cms.PSet (

  UseDuplicateRemoval = cms.bool   ( True ), # use tracklet seed type as channel id if False, binned track pt used if True
  PtBoundaries        = cms.vdouble( 1.34 ), # positive pt Boundaries in GeV (symmetric negatives are assumed), last boundary is infinity, defining ot bins used by DR

  SeedTypesReduced = cms.vstring( "L1L2", "L2L3", "L3L4", "L5L6", "D1D2", "D3D4", "L1D1", "L2D1" ), # seed types used in reduced tracklet algorithm (position gives int value)

  MaxNumProjectionLayers     = cms.int32( 8 ), # max number layers a sedd type may project to
  SeedTypesSeedLayersReduced = cms.PSet (      # seeding layers of seed types in reduced config using default layer id [barrel: 1-6, discs: 11-15]
    L1L2 = cms.vint32(  1,  2 ),
    L2L3 = cms.vint32(  2,  3 ),
    L3L4 = cms.vint32(  3,  4 ),
    L5L6 = cms.vint32(  5,  6 ),
    D1D2 = cms.vint32( 11, 12 ),
    D3D4 = cms.vint32( 13, 14 ),
    L1D1 = cms.vint32(  1, 11 ),
    L2D1 = cms.vint32(  2, 11 )
  ),
  SeedTypesProjectionLayersReduced = cms.PSet ( # layers a seed types can project to in reduced config using default layer id [barrel: 1-6, discs: 11-15]
    L1L2 = cms.vint32(  3,  4,  5,  6, 11, 12, 13, 14 ),
    L2L3 = cms.vint32(  1,  4,  5,  6, 11, 12, 13, 14 ),
    L3L4 = cms.vint32(  1,  2,  5,  6, 11, 12 ),
    L5L6 = cms.vint32(  1,  2,  3,  4 ),
    D1D2 = cms.vint32(  1,  2, 13, 14, 15 ),
    D3D4 = cms.vint32(  1, 11, 12, 15 ),
    L1D1 = cms.vint32( 12, 13, 14, 15 ),
    L2D1 = cms.vint32(  1, 12, 13, 14 )
  ),

  IRChannelsIn = cms.vint32( range(0, 48) ) # map of used DTC tfp channels in InputRouter

)