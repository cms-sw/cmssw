# The following comments couldn't be translated into the new config version:

# histos limits and binning

import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmElectronTagProbeAnalysis = DQMEDAnalyzer('ElectronTagProbeAnalyzer',

    Verbosity = cms.untracked.int32(0),
    FinalStep = cms.string("AtJobEnd"),
    InputFile = cms.string(""),
    OutputFile = cms.string(""),
    InputFolderName = cms.string("Egamma/Electrons/TagAndProbe"),
    OutputFolderName = cms.string("Egamma/Electrons/TagAndProbe"),
    
    Selection = cms.int32(3), # 0=All elec, 1=Etcut, 2=Iso, 3=eId
    ElectronCollection = cms.InputTag("gedGsfElectrons"),
    MatchingObjectCollection = cms.InputTag("mergedSuperClusters"),
    TrackCollection = cms.InputTag("generalTracks"),
    GsfTrackCollection = cms.InputTag("electronGsfTracks"),
    VertexCollection = cms.InputTag(""),
    BeamSpot = cms.InputTag("offlineBeamSpot"),
    ReadAOD = cms.bool(False),
    
    #MatchingCondition = cms.string("Cone"),
    #MaxPtMatchingObject = cms.double(100.0),
    #MaxAbsEtaMatchingObject = cms.double(2.5),
    #DeltaR = cms.double(0.3),
    
    MassLow = cms.double(60),
    MassHigh = cms.double(120),
    TpCheckSign = cms.bool(True),  
    TagCheckClass = cms.bool(False),  
    ProbeEtCut = cms.bool(False),
    ProbeCheckClass = cms.bool(False),                                        

    MinEt = cms.double(10.),
    MinPt = cms.double(0.),
    MaxAbsEta = cms.double(2.5),
    SelectEb = cms.bool(False),
    SelectEe = cms.bool(False),
    SelectNotEbEeGap = cms.bool(False),
    SelectEcalDriven = cms.bool(False),
    SelectTrackerDriven = cms.bool(False),
    MinEopBarrel = cms.double(0.),
    MaxEopBarrel = cms.double(10000.),
    MinEopEndcaps = cms.double(0.),
    MaxEopEndcaps = cms.double(10000.),
    MinDetaBarrel = cms.double(0.),
    MaxDetaBarrel = cms.double(10000.),
    MinDetaEndcaps = cms.double(0.),
    MaxDetaEndcaps = cms.double(10000.),
    MinDphiBarrel = cms.double(0.),
    MaxDphiBarrel = cms.double(10000.),
    MinDphiEndcaps = cms.double(0.),
    MaxDphiEndcaps = cms.double(10000.),
    MinSigIetaIetaBarrel = cms.double(0.),
    MaxSigIetaIetaBarrel = cms.double(10000.),
    MinSigIetaIetaEndcaps = cms.double(0.),
    MaxSigIetaIetaEndcaps = cms.double(10000.),
    MaxHoeBarrel = cms.double(10000.),
    MaxHoeEndcaps = cms.double(10000.),
    MinMva = cms.double(-10000.),
    MaxTipBarrel = cms.double(10000.),
    MaxTipEndcaps = cms.double(10000.),
    MaxTkIso03 = cms.double(1.),
    MaxHcalIso03Depth1Barrel = cms.double(10000.),
    MaxHcalIso03Depth1Endcaps = cms.double(10000.),
    MaxHcalIso03Depth2Endcaps = cms.double(10000.),
    MaxEcalIso03Barrel = cms.double(10000.),
    MaxEcalIso03Endcaps = cms.double(10000.),

    TriggerResults = cms.InputTag("TriggerResults::HLT"),
    NbinEta = cms.int32(50), NbinEta2D = cms.int32(50), EtaMin = cms.double(-2.5), EtaMax = cms.double(2.5),
    NbinPhi = cms.int32(64), NbinPhi2D = cms.int32(32), PhiMax = cms.double(3.2), PhiMin = cms.double(-3.2),
    NbinPt = cms.int32(50), NbinPtEff = cms.int32(19), NbinPt2D = cms.int32(50), PtMax = cms.double(100.0),
    NbinP = cms.int32(50), NbinP2D = cms.int32(50), PMax = cms.double(300.0),
    NbinEop = cms.int32(50), NbinEop2D = cms.int32(30), EopMax = cms.double(5.0), EopMaxSht = cms.double(3.0),
    NbinDeta = cms.int32(100), DetaMin = cms.double(-0.005), DetaMax = cms.double(0.005),
    NbinDphi = cms.int32(100), DphiMin = cms.double(-0.01), DphiMax = cms.double(0.01),
    NbinDetaMatch = cms.int32(100), NbinDetaMatch2D = cms.int32(50), DetaMatchMin = cms.double(-0.05), DetaMatchMax = cms.double(0.05),
    NbinDphiMatch = cms.int32(100), NbinDphiMatch2D = cms.int32(50), DphiMatchMin = cms.double(-0.2), DphiMatchMax = cms.double(0.2),
    NbinFhits = cms.int32(30), FhitsMax = cms.double(30.0),
    NbinLhits = cms.int32(5), LhitsMax = cms.double(10.0),
    NbinXyz = cms.int32(50), NbinXyz2D = cms.int32(25),
    NbinPopTrue = cms.int32(75), PopTrueMin = cms.double(0.0), PopTrueMax = cms.double(1.5),
    NbinMee = cms.int32(100), MeeMin = cms.double(0.0), MeeMax = cms.double(150.),
    NbinHoe = cms.int32(100), HoeMin = cms.double(0.0), HoeMax = cms.double(0.5)
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( dqmElectronTagProbeAnalysis, ElectronCollection = cms.InputTag("ecalDrivenGsfElectrons") )
