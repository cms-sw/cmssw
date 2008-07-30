import FWCore.ParameterSet.Config as cms

SiStripMonitorDigi = cms.EDFilter("SiStripMonitorDigi",
    # by default do not write out any file with histograms
    # can overwrite this in .cfg file with: replace SiStripMonitorDigi.OutputMEsInRootFile = true
    OutputMEsInRootFile = cms.bool(False),
    # add digi producers same way as Domenico in SiStripClusterizer
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('ZeroSuppressed'),
        DigiProducer = cms.string('siStripDigis')
    ), 
        cms.PSet(
            DigiLabel = cms.string('VirginRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ProcessedRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ScopeMode'),
            DigiProducer = cms.string('siStripZeroSuppression')
        )),
    OutputFileName = cms.string('test_digi.root'),
    # rest of parameters
    SelectAllDetectors = cms.bool(False),
    CalculateStripOccupancy = cms.bool(False),
    ResetMEsEachRun = cms.bool(False),
    ShowMechanicalStructureView = cms.bool(True),
    ShowControlView = cms.bool(False),
    ShowReadoutView = cms.bool(False)
)



