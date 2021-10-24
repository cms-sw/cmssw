import FWCore.ParameterSet.Config as cms

TrackBuilderChannel_params = cms.PSet (

    UseDuplicateRemoval = cms.bool   ( True ),                           # use tracklet seed type as channel id if False, binned track pt used if True
    NumSeedTypes        = cms.int32  ( 8    ),                           # number of used seed types in tracklet algorithm
    #PtBoundaries        = cms.vdouble( 1.8, 2.16, 2.7, 3.6, 5.4, 10.8 ), # pt Boundaries in GeV, last boundary is infinity
    PtBoundaries        = cms.vdouble( 1.34 ), # pt Boundaries in GeV, last boundary is infinity

)