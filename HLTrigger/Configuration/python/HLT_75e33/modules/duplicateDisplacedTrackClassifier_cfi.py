import FWCore.ParameterSet.Config as cms

duplicateDisplacedTrackClassifier = cms.EDProducer("TrackCutClassifier",
    beamspot = cms.InputTag("offlineBeamSpot"),
    ignoreVertices = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    mva = cms.PSet(
        dr_par = cms.PSet(
            d0err = cms.vdouble(0.003, 0.003, 0.003),
            d0err_par = cms.vdouble(0.001, 0.001, 0.001),
            drWPVerr_par = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38),
            dr_exp = cms.vint32(2147483647, 2147483647, 2147483647),
            dr_par1 = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38),
            dr_par2 = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38)
        ),
        dz_par = cms.PSet(
            dzWPVerr_par = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38),
            dz_exp = cms.vint32(2147483647, 2147483647, 2147483647),
            dz_par1 = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38),
            dz_par2 = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38)
        ),
        isHLT = cms.bool(False),
        maxChi2 = cms.vdouble(9999.0, 9999.0, 9999.0),
        maxChi2n = cms.vdouble(9999.0, 9999.0, 9999.0),
        maxDr = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38),
        maxDz = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38),
        maxDzWrtBS = cms.vdouble(3.40282346639e+38, 24, 15),
        maxLostLayers = cms.vint32(99, 99, 99),
        maxRelPtErr = cms.vdouble(3.40282346639e+38, 3.40282346639e+38, 3.40282346639e+38),
        min3DLayers = cms.vint32(0, 0, 0),
        minHits = cms.vint32(0, 0, 1),
        minHits4pass = cms.vint32(2147483647, 2147483647, 2147483647),
        minLayers = cms.vint32(0, 0, 0),
        minNVtxTrk = cms.int32(2),
        minNdof = cms.vdouble(-1, -1, -1),
        minPixelHits = cms.vint32(0, 0, 0)
    ),
    qualityCuts = cms.vdouble(-0.7, 0.1, 0.7),
    src = cms.InputTag("mergedDuplicateDisplacedTracks"),
    vertices = cms.InputTag("firstStepPrimaryVertices")
)
