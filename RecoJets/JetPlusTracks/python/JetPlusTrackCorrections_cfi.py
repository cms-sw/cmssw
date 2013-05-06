import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import * 

JPTZSPCorrectorICone5 = cms.PSet(

    # General Configuration
    Verbose = cms.bool(True),

    # Vectorial corrections
    VectorialCorrection  = cms.bool(True),
    UseResponseInVecCorr = cms.bool(False),
    
    # Select tracks used in correction
    UseInConeTracks      = cms.bool(True),
    UseOutOfConeTracks   = cms.bool(True),
    UseOutOfVertexTracks = cms.bool(True),
    
    # Jet-tracks association
    JetTracksAssociationAtVertex   = cms.InputTag("iterativeCone5JetTracksAssociatorAtVertexJPT"), 
    JetTracksAssociationAtCaloFace = cms.InputTag("iterativeCone5JetTracksAssociatorAtCaloFace"),

    # Jet merging/splitting
    JetSplitMerge = cms.int32(0),

    # Pions
    UsePions      = cms.bool(True),
    UseEfficiency = cms.bool(True),
    
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
    PtErrorQuality  = cms.double(0.05),
    DzVertexCut     = cms.double(0.2),
    
    # Response and efficiency maps
    ResponseMap   = cms.string("CondFormats/JetMETObjects/data/CMSSW_538_response.txt"),
    EfficiencyMap = cms.string("CondFormats/JetMETObjects/data/CMSSW_538_TrackNonEff.txt"),
    LeakageMap    = cms.string("CondFormats/JetMETObjects/data/CMSSW_538_TrackLeakage.txt"),

    # Jet-related
    MaxJetEta = cms.double(3.0)
    
    )
