from __future__ import print_function
import FWCore.ParameterSet.Config as cms
process = cms.Process("MCMisalignmentScaler")

process.load("Configuration.StandardSequences.MagneticField_cff") # B-field map
process.load("Configuration.Geometry.GeometryRecoDB_cff") # Ideal geometry and interface
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") # Global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet(record = cms.string("RunInfoRcd"),           tag = cms.string("")),
    cms.PSet(record = cms.string("SiStripBadChannelRcd"), tag = cms.string("")),
    cms.PSet(record = cms.string("SiStripBadFiberRcd"),   tag = cms.string("")),
    cms.PSet(record = cms.string("SiStripBadModuleRcd"),  tag = cms.string("")),
    cms.PSet(record = cms.string("SiStripBadStripRcd"),   tag = cms.string("")),
    cms.PSet(record = cms.string("SiStripDetCablingRcd"), tag = cms.string("")),
)
process.load("Alignment.TrackerAlignment.mcMisalignmentScaler_cfi")

################################################################################
# parameters to configure:
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2017_realistic")
process.mcMisalignmentScaler.scalers.append(
    cms.PSet(
        subDetector = cms.untracked.string("Tracker"),
        factor = cms.untracked.double(0.2)
    )
)
process.mcMisalignmentScaler.pullBadModulesToIdeal = False
process.mcMisalignmentScaler.outlierPullToIdealCut = 0.1
################################################################################


usedGlobalTag = process.GlobalTag.globaltag.value()
print("Using Global Tag:", usedGlobalTag)

from CondCore.CondDB.CondDB_cfi import CondDB
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDB,
    timetype = cms.untracked.string("runnumber"),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string("TrackerAlignmentRcd"),
            tag = cms.string("Alignments")
        ),
    )
)
process.PoolDBOutputService.connect = "sqlite_file:misalignment_rescaled0p2.db"

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")
process.p = cms.Path(process.mcMisalignmentScaler)
