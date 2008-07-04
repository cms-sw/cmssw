import FWCore.ParameterSet.Config as cms

# SiStripMonitorCluster
SiStripMonitorCluster = cms.EDFilter("SiStripMonitorCluster",
    # by default do not write out any file with histograms
    # can overwrite this in .cfg file with: replace SiStripMonitorCluster.OutputMEsInRootFile = true
    OutputMEsInRootFile = cms.bool(False),
    ClusterLabel = cms.string(''),
    OutputFileName = cms.string('test_digi_cluster.root'),
    #
    SelectAllDetectors = cms.bool(False),
    ClusterProducer = cms.string('siStripClusters'),
    FillSignalNoiseHistos = cms.bool(True),
    ShowMechanicalStructureView = cms.bool(True),
    ShowControlView = cms.bool(False),
    ResetMEsEachRun = cms.bool(False),
    ShowReadoutView = cms.bool(False)
)


