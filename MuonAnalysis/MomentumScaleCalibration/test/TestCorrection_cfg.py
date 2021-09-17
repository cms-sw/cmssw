import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTCORRECTION")
process.load("MuonAnalysis.MomentumScaleCalibration.local_CSA08_Y_cff")

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
    input = cms.untracked.int32(1)
)
process.TestCorrectionModule = cms.EDAnalyzer(
    "TestCorrection",
    process.MuonServiceProxy,
    # // standalone muons
    # InputTag MuonLabel = standAloneMuons:UpdatedAtVtx
    # int32 muonType = 2 
    # // tracker tracks
    # InputTag MuonLabel = generalTracks //ctfWithMaterialTracks
    # int32 muonType = 3 
    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    OutputFileName = cms.untracked.string('TestCorrection.root'),
    # Choose the kind of muons you want to run on
    # -------------------------------------------
    # global muons
    MuonType = cms.int32(1),
    MuonLabel = cms.InputTag("muons"),

    # Specify the corrections to use
    CorrectionsIdentifier = cms.untracked.string('MCcorrDerivedFromY_globalMuons_test')
)

process.p1 = cms.Path(process.TestCorrectionModule)

# Timing information
process.load("FWCore.MessageLogger.MessageLogger_cfi")
TimingLogFile = cms.untracked.string('timing.log')
process.Timing = cms.Service("Timing")

