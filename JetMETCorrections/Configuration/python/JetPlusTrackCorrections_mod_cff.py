import FWCore.ParameterSet.Config as cms

# "Generic" configurables used by ESSources/EDProducers in both the JetMET and PAT code 

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * 

JPTZSPCorrectorICone5 = cms.PSet(

    # General Configuration
    Verbose           = cms.bool(False),
    UsePatCollections = cms.bool(False),
    
    # Select correction types
    UseInConeTracks      = cms.bool(True),
    UseOutOfConeTracks   = cms.bool(True),
    UseOutOfVertexTracks = cms.bool(True),
    
    # Jet-tracks association (null value = "on-the-fly" mode)
    JetTracksAssociationAtVertex   = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtVertex"), 
    JetTracksAssociationAtCaloFace = cms.InputTag("ZSPiterativeCone5JetTracksAssociatorAtCaloFace"),

    # Jet merging/splitting
    JetSplitMerge = cms.int32(0),
    
    # Jet-tracks association "on-the-fly" mode
    AllowOnTheFly = cms.bool(False),
    Tracks        = cms.InputTag("generalTracks"),
    Propagator    = cms.string('SteppingHelixPropagatorAlong'),
    ConeSize      = cms.double(0.5),
    
    # Muons
    UseMuons = cms.bool(True),
    Muons    = cms.InputTag("muons"),
    
    # Electrons
    UseElectrons    = cms.bool(True),
    Electrons       = cms.InputTag("gsfElectrons"),
    ElectronIds     = cms.InputTag("JPTeidTight"),
    
    # Filtering tracks using quality
    UseTrackQuality = cms.bool(True),
    TrackQuality    = cms.string('highPurity'),
    
    # Response and efficiency maps
    ResponseMap   = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_response.txt"),
    EfficiencyMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackNonEff.txt"),
    LeakageMap    = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackLeakage.txt"),

    )
JPTZSPCorrectorSisCone5 = cms.PSet(

    # General Configuration
    Verbose           = cms.bool(False),
    UsePatCollections = cms.bool(False),

    # Select correction types
    UseInConeTracks      = cms.bool(True),
    UseOutOfConeTracks   = cms.bool(True),
    UseOutOfVertexTracks = cms.bool(True),

    # Jet-tracks association (null value = "on-the-fly" mode)
    JetTracksAssociationAtVertex   = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtVertex"),
    JetTracksAssociationAtCaloFace = cms.InputTag("ZSPSisCone5JetTracksAssociatorAtCaloFace"),

    # Jet merging/splitting
    JetSplitMerge = cms.int32(1),

    # Jet-tracks association "on-the-fly" mode
    AllowOnTheFly = cms.bool(False),
    Tracks        = cms.InputTag("generalTracks"),
    Propagator    = cms.string('SteppingHelixPropagatorAlong'),
    ConeSize      = cms.double(0.5),

    # Muons
    UseMuons = cms.bool(True),
    Muons    = cms.InputTag("muons"),

    # Electrons
    UseElectrons    = cms.bool(True),
    Electrons       = cms.InputTag("gsfElectrons"),
    ElectronIds     = cms.InputTag("JPTeidTight"),

    # Filtering tracks using quality
    UseTrackQuality = cms.bool(True),
    TrackQuality    = cms.string('highPurity'),

    # Response and efficiency maps
    ResponseMap   = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_response.txt"),
    EfficiencyMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackNonEff.txt"),
    LeakageMap    = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackLeakage.txt"),

    )
JPTZSPCorrectorAntiKt5 = cms.PSet(

    # General Configuration
    Verbose           = cms.bool(False),
    UsePatCollections = cms.bool(False),

    # Select correction types
    UseInConeTracks      = cms.bool(True),
    UseOutOfConeTracks   = cms.bool(True),
    UseOutOfVertexTracks = cms.bool(True),

    # Jet-tracks association (null value = "on-the-fly" mode)
    JetTracksAssociationAtVertex   = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtVertex"),
    JetTracksAssociationAtCaloFace = cms.InputTag("ZSPAntiKt5JetTracksAssociatorAtCaloFace"),

    # Jet merging/splitting
    JetSplitMerge = cms.int32(2),

    # Jet-tracks association "on-the-fly" mode
    AllowOnTheFly = cms.bool(False),
    Tracks        = cms.InputTag("generalTracks"),
    Propagator    = cms.string('SteppingHelixPropagatorAlong'),
    ConeSize      = cms.double(0.5),

    # Muons
    UseMuons = cms.bool(True),
    Muons    = cms.InputTag("muons"),

    # Electrons
    UseElectrons    = cms.bool(True),
    Electrons       = cms.InputTag("gsfElectrons"),
    ElectronIds     = cms.InputTag("JPTeidTight"),

    # Filtering tracks using quality
    UseTrackQuality = cms.bool(True),
    TrackQuality    = cms.string('highPurity'),

    # Response and efficiency maps
    ResponseMap   = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_response.txt"),
    EfficiencyMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackNonEff.txt"),
    LeakageMap    = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackLeakage.txt"),

    )

