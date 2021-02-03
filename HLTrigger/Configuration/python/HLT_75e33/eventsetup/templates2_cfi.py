import FWCore.ParameterSet.Config as cms

templates2 = cms.ESProducer("PixelCPEClusterRepairESProducer",
    Alpha2Order = cms.bool(True),
    ClusterProbComputationFlag = cms.int32(0),
    ComponentName = cms.string('PixelCPEClusterRepair'),
    DoLorentz = cms.bool(True),
    LoadTemplatesFromDB = cms.bool(True),
    MaxSizeMismatchInY = cms.double(0.3),
    MinChargeRatio = cms.double(0.8),
    Recommend2D = cms.vstring(
        'PXB 2',
        'PXB 3',
        'PXB 4'
    ),
    RunDamagedClusters = cms.bool(False),
    UseClusterSplitter = cms.bool(False),
    appendToDataLabel = cms.string(''),
    barrelTemplateID = cms.int32(0),
    directoryWithTemplates = cms.int32(0),
    forwardTemplateID = cms.int32(0),
    lAOffset = cms.double(0),
    lAWidthBPix = cms.double(0),
    lAWidthFPix = cms.double(0),
    speed = cms.int32(-2),
    useLAWidthFromDB = cms.bool(True)
)
