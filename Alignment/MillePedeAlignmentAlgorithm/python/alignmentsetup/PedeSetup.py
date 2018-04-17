import FWCore.ParameterSet.Config as cms
import Alignment.MillePedeAlignmentAlgorithm.mpslib.tools as mps_tools


def setup(process, binary_files, tree_files, run_start_geometry):
    """Pede-specific setup.

    Arguments:
    - `process`: cms.Process object
    - `binary_files`: list of binary files to be read by pede
    - `tree_files`: list of ROOT files created in the mille step
    - `run_start_geometry`: run ID to pick the start geometry
    """

    # write alignments, APEs, and surface deformations to DB by default
    # --------------------------------------------------------------------------
    process.AlignmentProducer.saveToDB = True
    process.AlignmentProducer.saveApeToDB = True
    process.AlignmentProducer.saveDeformationsToDB = True

    # setup database output module
    # --------------------------------------------------------------------------
    from CondCore.CondDB.CondDB_cfi import CondDB
    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
        CondDB.clone(connect = "sqlite_file:alignments_MP.db"),
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
                record = cms.string("SiStripLorentzAngleRcd_peak"),
                tag = cms.string("SiStripLorentzAngle_peak")),
            cms.PSet(
                record = cms.string("SiStripLorentzAngleRcd_deco"),
                tag = cms.string("SiStripLorentzAngle_deco")),
            cms.PSet(
                record = cms.string("SiPixelLorentzAngleRcd"),
                tag = cms.string("SiPixelLorentzAngle")),
            cms.PSet(
                record = cms.string("SiStripBackPlaneCorrectionRcd"),
                tag = cms.string("SiStripBackPlaneCorrection"))
        )
    )


    # Reconfigure parts of the algorithm configuration
    # --------------------------------------------------------------------------
    process.AlignmentProducer.algoConfig.mergeBinaryFiles = binary_files
    process.AlignmentProducer.algoConfig.mergeTreeFiles   = tree_files


    # align calibrations to general settings
    # --------------------------------------------------------------------------
    for calib in process.AlignmentProducer.calibrations:
        calib.saveToDB       = process.AlignmentProducer.saveToDB
        calib.treeFile       = process.AlignmentProducer.algoConfig.treeFile
        calib.mergeTreeFiles = process.AlignmentProducer.algoConfig.mergeTreeFiles


    # Configure the empty source to include all needed runs
    # --------------------------------------------------------------------------
    iovs = mps_tools.make_unique_runranges(process.AlignmentProducer)
    number_of_events = iovs[-1] - iovs[0] + 1

    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(number_of_events))
    process.source = cms.Source(
        "EmptySource",
        firstRun = cms.untracked.uint32(run_start_geometry),
        numberEventsInRun = cms.untracked.uint32(1))

    # Define the executed path
    # --------------------------------------------------------------------------
    process.p = cms.Path(process.AlignmentProducer)
