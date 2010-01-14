import FWCore.ParameterSet.Config as cms

TrackJetParameters = cms.PSet(
    src            = cms.InputTag('trackRefsForJets'),
    srcPVs         = cms.InputTag('offlinePrimaryVertices'),
    jetType        = cms.string('TrackJet'),
    jetPtMin       = cms.double(0.3),
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
      # if doPU is false, these are not read:
      Active_Area_Repeats = cms.int32(1),
      GhostArea = cms.double(0.01),
      Ghost_EtaMax = cms.double(5.0),
    # only use the tracks that were used to fit the vertex
    UseOnlyVertexTracks = cms.bool(False),
    # only consider the highest-sum-pT PV for clustering
    UseOnlyOnePV        = cms.bool(True),
    # maximum z-distance between track and vertex for association (in cm)
    DzTrVtxMax          = cms.double(0.5),
    # maximum xy-distance between track and vertex for association (in cm)
    DxyTrVtxMax         = cms.double(0.1)

    )
