import FWCore.ParameterSet.Config as cms

trackingCertificationInfo = cms.EDAnalyzer("TrackingCertificationInfo",
    TopFolderName = cms.untracked.string("Tracking"),
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
    ),
)
