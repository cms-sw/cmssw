import FWCore.ParameterSet.Config as cms

SiStripMonitorFilter = cms.EDAnalyzer("SiStripMonitorFilter",
    # by default do not write out any file with histograms
    # can overwrite this in .cfg file with: replace SiStripMonitorFilter.OutputMEsInRootFile = true
    OutputMEsInRootFile = cms.bool(False),
    FilterProducer = cms.string('ClusterMTCCFilter'),
    OutputFileName = cms.string('dqm_sistrip_filter.root')
)


