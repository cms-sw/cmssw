import FWCore.ParameterSet.Config as cms

l1tTrackerEmuEtMiss = cms.EDProducer('L1TrackerEtMissEmulatorProducer',
    L1TrackInputTag =  cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelectedEmulation"),
    L1TrackAssociatedInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForEtMiss", "Level1TTTracksSelectedAssociatedEmulation"),
    # To bypass GTT input module use  cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks")
    # and set useGTTinput to false
    L1VertexInputTag = cms.InputTag("l1tVertexFinderEmulator", "l1verticesEmulation"),
    # This will use the vertex algorithm as specified in l1tVertexProducer_cfi, if using emulated vertex
    # set useVertexEmulator to true
    L1MetCollectionName = cms.string("L1TrackerEmuEtMiss"),
    
    nCordicSteps = cms.int32( 13 ), #Number of steps for cordic sqrt and phi computation
    debug        = cms.int32( 0 ),  #0 - No Debug, 1 - LUT debug, 2 - Phi Debug, 3 - Z debug, 4 - Et Debug, 5 - Cordic Debug, 6 - Output, 7 - Every Selected Track
    useGTTinput  = cms.bool( True ),

)

