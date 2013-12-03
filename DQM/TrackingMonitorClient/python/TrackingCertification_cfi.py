import FWCore.ParameterSet.Config as cms

trackingCertificationInfo = cms.EDAnalyzer("TrackingCertificationInfo",
    TopFolderName = cms.untracked.string("Tracking"),
    checkPixelFEDs = cms.bool(False),
    TrackingGlobalQualityPSets = cms.VPSet(
         cms.PSet(
             QT       = cms.string("Rate"),
         ),
         cms.PSet(
             QT       = cms.string("Chi2"),
         ),
         cms.PSet(
             QT       = cms.string("RecHits"),
         ),
         cms.PSet(
             QT       = cms.string("Seed"),
         ),
    ),
    TrackingLSQualityMEs = cms.VPSet(
         cms.PSet(
             QT       = cms.string("Rate"),
         ),
         cms.PSet(
             QT       = cms.string("Chi2"),
         ),
         cms.PSet(
             QT       = cms.string("RecHits"),
         ),
         cms.PSet(
             QT       = cms.string("Seed"),
         ),
    ),
)
