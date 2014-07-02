import FWCore.ParameterSet.Config as cms
cleaningParameters = cms.PSet(

    gtLabel      = cms.InputTag("gtDigis"),
  
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
    
    #Turn on extra checks
    bypassAllPVChecks   = cms.bool(False),
    bypassAllDCSChecks   = cms.bool(False),
    vertexCollection    = cms.InputTag( "goodOfflinePrimaryVerticesDQM" ), #From CommonTools/ParticleFlow/goodOfflinePrimaryVertices_cfi.py
#    doHLTPhysicsOn      = cms.bool(False),
 
)

