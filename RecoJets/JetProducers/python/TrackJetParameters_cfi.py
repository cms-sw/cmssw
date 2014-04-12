import FWCore.ParameterSet.Config as cms

TrackJetParameters = cms.PSet(
    src            = cms.InputTag('trackRefsForJets'),
    srcPVs         = cms.InputTag('offlinePrimaryVertices'),
    jetType        = cms.string('TrackJet'),
    doOutputJets   = cms.bool(True),
    jetPtMin       = cms.double(3.0),
    inputEMin      = cms.double(0.0),
    inputEtMin     = cms.double(0.0),
    doPVCorrection = cms.bool(False),
    # pileup with offset correction
    doPUOffsetCorr = cms.bool(False),
    # if pileup is false, these are not read:
    nSigmaPU = cms.double(1.0),
    radiusPU = cms.double(0.5),  
    # fastjet-style pileup     
    doAreaFastjet   = cms.bool(False),
    doRhoFastjet    = cms.bool(False),
    doAreaDiskApprox= cms.bool( False),
    voronoiRfact    = cms.double(-0.9),
    # if doPU is false, these are not read:
    Active_Area_Repeats = cms.int32(1),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(5.0),
    # only use the tracks that were used to fit the vertex
    UseOnlyVertexTracks = cms.bool(False),
    # only consider the highest-sum-pT PV for clustering
    UseOnlyOnePV        = cms.bool(False),
    # maximum z-distance between track and vertex for association (in cm)
    DzTrVtxMax          = cms.double(1),
    # maximum xy-distance between track and vertex for association (in cm)
    DxyTrVtxMax         = cms.double(0.2),
    # minimum number of degrees of freedom to call a PV a good vertex
    MinVtxNdof          = cms.int32(5),
    # maximum z distance to origin to call a PV a good vertex
    MaxVtxZ             = cms.double(15.),
    useDeterministicSeed= cms.bool( True ),
    minSeed             = cms.uint32( 14327 )
)
