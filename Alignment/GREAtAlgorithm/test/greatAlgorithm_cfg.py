import FWCore.ParameterSet.Config as cms
process = cms.Process("Alignment")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") # Global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.load("Alignment.GREAtAlgorithm.greatAlgorithm_cfi")

################################################################################
# parameters to configure:
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:run2_data")

process.greatAlgorithm.inverseLevelOfTrustworthiness = 1.0
################################################################################

from CondCore.CondDB.CondDB_cfi import CondDB
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondDB.clone(connect = "sqlite_file:great_alignments.db"),
    timetype = cms.untracked.string("runnumber"),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string("TrackerAlignmentRcd"),
            tag = cms.string("Alignments")),
        cms.PSet(
            record = cms.string("TrackerAlignmentErrorExtendedRcd"),
            tag = cms.string("AlignmentErrorsExtended")),
        cms.PSet(
            record = cms.string("TrackerSurfaceDeformationRcd"),
            tag = cms.string("Deformations")),
        cms.PSet(
            record = cms.string("SiPixelLorentzAngleRcd"),
            tag = cms.string("SiPixelLorentzAngle")),
        cms.PSet(
            record = cms.string("SiStripLorentzAngleRcd"),
            tag = cms.string("SiStripLorentzAngle_"+process.greatAlgorithm.apvMode.value())),
        cms.PSet(
            record = cms.string("SiStripBackPlaneCorrectionRcd"),
            tag = cms.string("SiStripBackPlaneCorrection_"+process.greatAlgorithm.apvMode.value()))
    )
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(20180401))

process.p = cms.Path(process.greatAlgorithm)

print "Using Global Tag:", process.GlobalTag.globaltag.value()
