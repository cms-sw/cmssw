import FWCore.ParameterSet.Config as cms

cleaningParameters = cms.PSet(

    vertexLabel  = cms.InputTag("offlinePrimaryVertices"),
    gtLabel      = cms.InputTag("gtDigis"),

    HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
    
    trigSelection = cms.PSet(
        andOr         = cms.bool( False ),
        #dbLabel       = cms.string( 'jetmet_trigsel' ), # will be discussed below (DB)
        #dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
        #dcsPartitions = cms.vint32( 24, 25, 26, 27 ),
        #andOrDcs      = cms.bool( False ),
        #errorReplyDcs = cms.bool( False ),
        #gtInputTag    = cms.InputTag( "gtDigis" ),
        #gtDBKey       = cms.string( 'jetmet_gtsel' ),
        #gtStatusBits  = cms.vstring( 'PhysDecl' ), 
        #andOrGt       = cms.bool( False ),
        #errorReplyGt  = cms.bool( False ),
        #l1DBKey       = cms.string( 'jetmet_l1sel' ),
        #l1Algorithms  = cms.vstring( 'L1Tech_BPTX_plus_AND_minus.v0 AND ( L1Tech_BSC_minBias_threshold1.v0 OR L1Tech_BSC_minBias_threshold2.v0 ) AND NOT ( L1Tech_BSC_halo_beam2_inner.v0 OR L1Tech_BSC_halo_beam2_outer.v0 OR L1Tech_BSC_halo_beam1_inner.v0 OR L1Tech_BSC_halo_beam1_outer.v0 )' ), 
        #andOrL1       = cms.bool( False ),
        #errorReplyL1  = cms.bool( False ),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        hltDBKey       = cms.string( 'jetmet_hltsel' ),
        hltPaths       = cms.vstring( '' ), 
        andOrHlt       = cms.bool( False ),
        errorReplyHlt  = cms.bool( False ),
    ),
    techTrigsAND    = cms.vuint32(),
    techTrigsOR     = cms.vuint32(),
    techTrigsNOT    = cms.vuint32(),
    
    #Turn on extra checks
    doPrimaryVertexCheck   = cms.bool(True),
    doHLTPhysicsOn         = cms.bool(False),
    
    #Vertex cleanup parameters
    nvtx_min       = cms.int32(1), 
    nvtxtrks_min   = cms.int32(0), #not used by default
    vtxndof_min    = cms.int32(4),
    vtxchi2_max    = cms.double(9999), #not used by default
    vtxz_max       = cms.double(24.0),
    
    #Switch on  tight filters for BeamHalo, JetID, HCALnoise
    tightBHFiltering    = cms.bool(False),
    tightJetIDFiltering = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
)
