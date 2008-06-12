import FWCore.ParameterSet.Config as cms

SiStripPartitions = cms.PSet(
    PrimaryPartition = cms.untracked.PSet(
        FineDelayVersion = cms.untracked.vuint32(0, 0),
        # Overrides run number AND versions 
        ForceCurrentState = cms.untracked.bool(False),
        InputDcuInfoXml = cms.untracked.string('/afs/cern.ch/cms/cmt/onlinedev/data/examples/dcuinfo.xml'),
        VpspScanVersion = cms.untracked.vuint32(0, 0),
        DcuDetIdsVersion = cms.untracked.vuint32(0, 0),
        PartitionName = cms.untracked.string(''),
        # Syntax is {major,minor}. {0,0} means "current state". ("history")
        FastCablingVersion = cms.untracked.vuint32(0, 0),
        # Null value means use (analysis) versions below ("calibration")
        GlobalAnalysisVersion = cms.untracked.uint32(0),
        ApvLatencyVersion = cms.untracked.vuint32(0, 0),
        # Null value means use "latest run"
        RunNumber = cms.untracked.uint32(0),
        PedestalsVersion = cms.untracked.vuint32(0, 0),
        # Syntax is {major,minor}. {0,0} means "current state". 
        CablingVersion = cms.untracked.vuint32(0, 0),
        DcuPsuMapVersion = cms.untracked.vuint32(0, 0),
        # Overrides run number and uses versions below
        ForceVersions = cms.untracked.bool(False),
        InputFedXml = cms.untracked.vstring(''),
        InputFecXml = cms.untracked.vstring(''),
        ApvTimingVersion = cms.untracked.vuint32(0, 0),
        OptoScanVersion = cms.untracked.vuint32(0, 0),
        FecVersion = cms.untracked.vuint32(0, 0),
        # XML input files
        InputModuleXml = cms.untracked.string('/afs/cern.ch/cms/cmt/onlinedev/data/examples/module.xml'),
        ApvCalibVersion = cms.untracked.vuint32(0, 0),
        FedVersion = cms.untracked.vuint32(0, 0)
    )
)

