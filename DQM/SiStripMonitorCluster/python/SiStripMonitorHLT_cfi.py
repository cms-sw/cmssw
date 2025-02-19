import FWCore.ParameterSet.Config as cms

SiStripMonitorHLT = cms.EDAnalyzer("SiStripMonitorHLT",
    # by default do not write out any file with histograms
    # can overwrite this in .cfg file with: replace SiStripMonitorHLT.OutputMEsInRootFile = true
    OutputMEsInRootFile = cms.bool(False),
    HLTProducer = cms.string('trigger'),
    OutputFileName = cms.string('test_digi_hlt.root')
)


