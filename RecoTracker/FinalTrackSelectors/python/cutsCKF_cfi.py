# The following comments couldn't be translated into the new config version:

# we want to keep all the tracks with many hits
# less hits but vertex association
# less hits but vertex association
# even less hits, but decent track
# end of cut sets
import FWCore.ParameterSet.Config as cms

cutsCKF = cms.EDProducer("TrackMultiSelector",
    src = cms.InputTag("ctfWithMaterialTracks"),
    beamspot = cms.PSet( ## same as RecoTracker/TkTrackingRegions/data/GlobalTrackingRegionFromBeamSpot.cfi

        src = cms.InputTag("offlineBeamSpot"),
        dzSigmas = cms.double(3.0), ## z window relative to the interaction point spread (given by the beamSpot)

        d0 = cms.double(0.2)
    ),
    vtxTracks = cms.uint32(3), ## at least 3 tracks

    vtxChi2Prob = cms.double(0.01), ## at least 1% chi2nprobability (if it has a chi2)

    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(False),
    vertices = cms.InputTag("pixelVertices"),
    vtxNumber = cms.int32(-1),
    copyExtras = cms.untracked.bool(True), ## set to false on AOD

    splitOutputs = cms.untracked.bool(False),
    cutSets = cms.VPSet(cms.PSet(
        pt = cms.vdouble(0.3, 999999.0),
        validLayers = cms.vuint32(10, 999999),
        d0Rel = cms.double(9999.0),
        lostHits = cms.vuint32(0, 999999),
        chi2n = cms.vdouble(0.0, 999999.0),
        dz = cms.double(5.0),
        dzRel = cms.double(9999.0),
        d0 = cms.double(1.0)
    ), 
        cms.PSet(
            pt = cms.vdouble(0.3, 999999.0),
            validLayers = cms.vuint32(8, 9),
            d0Rel = cms.double(10.0),
            lostHits = cms.vuint32(0, 999999),
            chi2n = cms.vdouble(0.0, 999999.0),
            dz = cms.double(2.0),
            dzRel = cms.double(10.0),
            d0 = cms.double(0.2)
        ), 
        cms.PSet(
            pt = cms.vdouble(0.3, 999999.0),
            validLayers = cms.vuint32(5, 7),
            d0Rel = cms.double(7.0),
            lostHits = cms.vuint32(0, 999999),
            chi2n = cms.vdouble(0.0, 999999.0),
            dz = cms.double(0.5),
            dzRel = cms.double(7.0),
            d0 = cms.double(0.04)
        ), 
        cms.PSet(
            pt = cms.vdouble(0.3, 999999.0),
            validLayers = cms.vuint32(3, 4),
            d0Rel = cms.double(3.0),
            lostHits = cms.vuint32(0, 0),
            chi2n = cms.vdouble(0.0, 100.0),
            dz = cms.double(0.2),
            dzRel = cms.double(3.0),
            d0 = cms.double(0.02)
        ))
)


