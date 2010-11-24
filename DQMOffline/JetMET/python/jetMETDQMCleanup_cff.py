import FWCore.ParameterSet.Config as cms

cleaningParameters = cms.PSet(

    vertexLabel  = cms.InputTag("offlinePrimaryVertices"),
    gtLabel      = cms.InputTag("gtDigis"),

    HLT_PhysDec   = cms.string("HLT_PhysicsDeclared"),
    
    techTrigsAND    = cms.vuint32(),
    techTrigsOR     = cms.vuint32(),
    techTrigsNOT    = cms.vuint32(),
    
    #Turn on extra checks
    doPrimaryVertexCheck   = cms.bool(True),
    doHLTPhysicsOn         = cms.bool(False),
    
    #Vertex cleanup parameters
    nvtx_min       = cms.int32(1), 
    nvtxtrks_min   = cms.int32(0), #not used by default
    vtxndof_min    = cms.int32(5),
    vtxchi2_max    = cms.double(9999), #not used by default
    vtxz_max       = cms.double(15.0),
    
    #Switch on  tight filters for BeamHalo, JetID, HCALnoise
    tightBHFiltering      = cms.bool(False),
    tightJetIDFiltering   = cms.int32(-1), #-1 off, 0 minimal, 1 loose, 2 tight
    tightHcalFiltering    = cms.bool(False)

)
