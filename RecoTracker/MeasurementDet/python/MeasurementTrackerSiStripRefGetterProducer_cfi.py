import FWCore.ParameterSet.Config as cms

measurementTrackerSiStripRefGetterProducer = cms.EDProducer("MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag("SiStripRawToClustersFacility"),
    measurementTrackerName = cms.string('MeasurementTrackerOnDemand')
)


