import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer
from L1Trigger.L1TTrackMatch.L1TrackSelectionProducer_cfi import L1TrackSelectionProducer

L1TrackerEmuEtMiss = cms.EDProducer('L1TrackerEtMissEmulatorProducer',
    L1TrackInputTag = cms.InputTag("L1TrackSelectionProducer", L1TrackSelectionProducer.outputCollectionName.value() + "Emulation"),
    L1TrackAssociatedInputTag = cms.InputTag("L1TrackSelectionProducer", L1TrackSelectionProducer.outputCollectionName.value() + "AssociatedEmulation"),
    # To bypass GTT input module use  cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks")
    # and set useGTTinput to false
    L1VertexInputTag = cms.InputTag("VertexProducer", VertexProducer.l1VertexCollectionName.value()),
    # This will use the vertex algorithm as specified in VertexProducer_cff, if using emulated vertex
    # set useVertexEmulator to true
    L1MetCollectionName = cms.string("L1TrackerEmuEtMiss"),
    
    nCordicSteps = cms.int32( 8 ), #Number of steps for cordic sqrt and phi computation
    debug        = cms.int32( 0 ),  #0 - No Debug, 1 - LUT debug, 2 - Phi Debug, 3 - Z debug, 4 - Et Debug, 5 - Cordic Debug, 6 - Output, 7 - Every Selected Track
    useGTTinput  = cms.bool( True ),

)

