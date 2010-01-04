import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * 

JPTZSPCorrectorICone5BG = cms.PSet(

    # General Configuration
    # Verbose = cms.bool(False),
   
    # Filtering tracks using quality

    UseTrackQuality = cms.bool(True),
    TrackQuality    = cms.string('highPurity'),
    tracks = cms.InputTag("hiGlobalPrimTracks"),
 
    # Response and efficiency maps
    ResponseMap   = cms.string("JetMETCorrections/Configuration/data/CMSSW_340_response"),
    EfficiencyMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_340_TrackNonEff"),
    LeakageMap    = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackLeakage"),
    
    )
