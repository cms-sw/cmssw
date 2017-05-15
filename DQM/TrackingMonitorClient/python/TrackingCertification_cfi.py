import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

trackingCertificationInfo = DQMEDHarvester("TrackingCertificationInfo",
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
