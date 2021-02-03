import FWCore.ParameterSet.Config as cms

HiTrackingRegionFactoryFromJetsBlock = cms.PSet(
    ComponentName = cms.string('TauRegionalPixelSeedGenerator'),
    RegionPSet = cms.PSet(
        JetSrc = cms.InputTag("iterativeConePu5CaloJets"),
        deltaEtaRegion = cms.double(0.1),
        deltaPhiRegion = cms.double(0.1),
        howToUseMeasurementTracker = cms.string('ForSiStrips'),
        measurementTrackerName = cms.InputTag("MeasurementTrackerEvent"),
        originHalfLength = cms.double(0.2),
        originRadius = cms.double(0.2),
        originZPos = cms.double(0.0),
        precise = cms.bool(True),
        ptMin = cms.double(5.0),
        vertexSrc = cms.InputTag("hiSelectedPixelVertex")
    )
)