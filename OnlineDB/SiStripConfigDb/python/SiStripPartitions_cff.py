import FWCore.ParameterSet.Config as cms

SiStripPartitions = cms.PSet(

    PrimaryPartition = cms.untracked.PSet(
    
        PartitionName = cms.untracked.string(''),

        # Overrides run number AND versions 
        ForceCurrentState = cms.untracked.bool(False),

        # Null value means use "latest run"
        RunNumber = cms.untracked.uint32(0),

        # Overrides run number and uses versions below
        ForceVersions = cms.untracked.bool(False),

        # Syntax is {major,minor}. (0,0) means "current state". 
        CablingVersion   = cms.untracked.vuint32(0,0),
        FecVersion       = cms.untracked.vuint32(0,0),
        FedVersion       = cms.untracked.vuint32(0,0),
        DcuDetIdsVersion = cms.untracked.vuint32(0,0),
        DcuPsuMapVersion = cms.untracked.vuint32(0,0),
        # Syntax is {major,minor}. (0,0) means no masking!
        MaskVersion      = cms.untracked.vuint32(0,0),

        # Null value means use (analysis) versions below ("calibration")
        GlobalAnalysisVersion = cms.untracked.uint32(0),

        # Syntax is {major,minor}. (0,0) means "current state". ("history")
        FastCablingVersion = cms.untracked.vuint32(0,0),
        ApvTimingVersion   = cms.untracked.vuint32(0,0),
        OptoScanVersion    = cms.untracked.vuint32(0,0),
        VpspScanVersion    = cms.untracked.vuint32(0,0),
        ApvCalibVersion    = cms.untracked.vuint32(0,0),
        PedestalsVersion   = cms.untracked.vuint32(0,0),
        ApvLatencyVersion  = cms.untracked.vuint32(0,0),
        FineDelayVersion   = cms.untracked.vuint32(0,0),

        # XML input files
        InputModuleXml  = cms.untracked.string('/afs/cern.ch/cms/cmt/onlinedev/data/examples/module.xml'),
        InputDcuInfoXml = cms.untracked.string('/afs/cern.ch/cms/cmt/onlinedev/data/examples/dcuinfo.xml'),
        InputFedXml     = cms.untracked.vstring(''),
        InputFecXml     = cms.untracked.vstring('')

    )

    # Additional partitions here...

)

