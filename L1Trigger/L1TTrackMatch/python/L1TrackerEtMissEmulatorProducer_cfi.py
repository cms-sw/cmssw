import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer

L1TrackerEmuEtMiss = cms.EDProducer('L1TrackerEtMissEmulatorProducer',
    L1TrackInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted"),
    # To bypass GTT input module use  cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks")
    # and set useGTTinput to false
    L1VertexInputTag = cms.InputTag("VertexProducer", VertexProducer.l1VertexCollectionName.value()),
    # This will use the vertex algorithm as specified in VertexProducer_cff, if using emulated vertex
    # set useVertexEmulator to true
    L1MetCollectionName = cms.string("L1TrackerEmuEtMiss"),

    maxZ0 = cms.double ( 15. ) ,    # in cm
    maxEta = cms.double ( 2.4 ) ,   # max eta allowed for chosen tracks
    minPt = cms.double( 2.02 ),
    chi2rzdofMax = cms.double( 5. ), # max chi2rz/dof allowed for chosen tracks
    chi2rphidofMax = cms.double( 20. ), # max chi2rphi/dof allowed for chosen tracks
    bendChi2Max = cms.double( 2.25 ),# max bendchi2 allowed for chosen tracks
    nStubsmin = cms.int32( 4 ),     # min number of stubs for the tracks

    z0Thresholds = cms.vdouble( 0.37, 0.5, 0.6, 0.75, 1.0, 1.6 ), # Threshold for track to vertex association.
    etaRegions = cms.vdouble( 0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4 ), # Eta bins for choosing deltaZ threshold.
    
    nCordicSteps = cms.int32( 8 ), #Number of steps for cordic sqrt and phi computation
    debug        = cms.int32( 0 ),  #0 - No Debug, 1 - LUT debug, 2 - Phi Debug, 3 - Z debug, 4 - Et Debug, 5 - Cordic Debug, 6 - Output, 7 - Every Selected Track
    useGTTinput  = cms.bool( True ),

)

