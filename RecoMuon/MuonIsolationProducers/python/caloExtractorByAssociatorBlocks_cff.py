import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.trackAssociatorBlocks_cff import *
MIsoCaloExtractorByAssociatorTowersBlock = cms.PSet(
    MIsoTrackAssociatorTowers,
    Noise_HE = cms.double(0.2),
    DR_Veto_H = cms.double(0.1),
    Noise_EE = cms.double(0.1),
    UseRecHitsFlag = cms.bool(False),
    NoiseTow_EE = cms.double(0.15),
    Threshold_HO = cms.double(0.5),
    Noise_EB = cms.double(0.025),
    Noise_HO = cms.double(0.2),
    CenterConeOnCalIntersection = cms.bool(False),
    DR_Max = cms.double(1.0),
    PropagatorName = cms.string('SteppingHelixPropagatorAny'),
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False )
    ),
    Threshold_E = cms.double(0.2),
    Noise_HB = cms.double(0.2),
    PrintTimeReport = cms.untracked.bool(False),
    NoiseTow_EB = cms.double(0.04),
    Threshold_H = cms.double(0.5),
    DR_Veto_E = cms.double(0.07),
    DepositLabel = cms.untracked.string('Cal'),
    ComponentName = cms.string('CaloExtractorByAssociator'),
    DR_Veto_HO = cms.double(0.1),
    DepositInstanceLabels = cms.vstring('ecal', 
        'hcal', 
        'ho')
)
MIsoCaloExtractorByAssociatorHitsBlock = cms.PSet(
    MIsoTrackAssociatorHits,
    Noise_HE = cms.double(0.2),
    DR_Veto_H = cms.double(0.1),
    Noise_EE = cms.double(0.1),
    UseRecHitsFlag = cms.bool(True),
    NoiseTow_EE = cms.double(0.15),
    Threshold_HO = cms.double(0.1),
    Noise_EB = cms.double(0.025),
    Noise_HO = cms.double(0.2),
    CenterConeOnCalIntersection = cms.bool(False),
    DR_Max = cms.double(1.0),
    PropagatorName = cms.string('SteppingHelixPropagatorAny'),
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False )
    ),
    Threshold_E = cms.double(0.025),
    Noise_HB = cms.double(0.2),
    NoiseTow_EB = cms.double(0.04),
    PrintTimeReport = cms.untracked.bool(False),
    Threshold_H = cms.double(0.1),
    DR_Veto_E = cms.double(0.07),
    DepositLabel = cms.untracked.string('Cal'),
    ComponentName = cms.string('CaloExtractorByAssociator'),
    DR_Veto_HO = cms.double(0.1),
    DepositInstanceLabels = cms.vstring('ecal', 
        'hcal', 
        'ho')
)


