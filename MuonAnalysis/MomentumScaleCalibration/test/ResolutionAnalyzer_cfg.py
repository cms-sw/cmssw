# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

process = cms.Process("RESOLUTIONANALYZER")
process.load("MuonAnalysis.MomentumScaleCalibration.local_CSA08_JPsi_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")

process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")

# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring()
# )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.ResolutionAnalyzerModule = cms.EDAnalyzer(
    "ResolutionAnalyzer",
    process.MuonServiceProxy,
    # // standalone muons
    # InputTag MuonLabel = standAloneMuons:UpdatedAtVtx
    # int32 muonType = 2 
    # // tracker tracks
    # InputTag MuonLabel = generalTracks //ctfWithMaterialTracks
    # int32 muonType = 3 

    # No fit is done, but the resolution function needs parameters
    # ResolFitType = cms.int32(6),
    # parResol = cms.vdouble(0.002, -0.0015, 0.000056, -0.00000085, 0.0000000046, -0.000027, 0.0037,
    #                        0.005, 0.00027, 0.0000027, 0.000094,
    #                        0.002, 0.00016, -0.00000051, 0.000022),

    # The eleven parResol parameters of resolfittype=8 are respectively:
    # constant of sigmaPt, Pt dep. of sigmaPt,
    # scale of the eta dep. made by points with values derived from MuonGun.
    # constant of sigmaCotgTheta, 1/Pt dep. of sigmaCotgTheta, Eta dep. of
    # sigmaCotgTheta, Eta^2 dep of sigmaCotgTheta;
    # constant of sigmaPhi, 1/Pt dep. of sigmaPhi, Eta dep. of sigmaPhi,
    # Eta^2 dep. of sigmaPhi.
    # This parameters are taken directly from the MuonGun (5<Pt<100, |eta|<3)
    ResolFitType = cms.int32(8),
    parResol = cms.vdouble(-0.007, 0.0001, 1.0,
                           0.00043, 0.0041, 0.0000028, 0.000077,
                           0.00011, 0.0018, -0.00000094, 0.000022),
    parResolFix = cms.vint32(0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0),
    parResolOrder = cms.vint32(0, 0, 0,
                               1, 1, 1, 1,
                               2, 2, 2, 2),


    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    ResFind = cms.vint32(0, 0, 0, 0, 0, 1),

    # Tree settings
    MaxEvents = cms.uint32(-1),
    InputTreeName = cms.string("tree.root"),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string('ResolutionAnalyzer_JPsi.root'),
    Debug = cms.untracked.bool(False),
    # Choose the kind of muons you want to run on
    # -------------------------------------------
    # global muons
    # MuonType = cms.int32(1),
    # MuonLabel = cms.InputTag("muons"),
    # // standalone muons
    # MuonType = cms.int32(2),
    # MuonLabel = cms.InputTag("standAloneMuons:UpdatedAtVtx"),
    # tracker tracks
    MuonType = cms.int32(3),
    MuonLabel = cms.InputTag("generalTracks"),
    Resonance = cms.untracked.bool(True),
    ReadCovariances = cms.untracked.bool(False),
    # This is used only when the ReadCovariances bool == True
    InputFileName = cms.untracked.string('ResolutionAnalyzer_JPsi.root')
)

process.p1 = cms.Path(process.ResolutionAnalyzerModule)

