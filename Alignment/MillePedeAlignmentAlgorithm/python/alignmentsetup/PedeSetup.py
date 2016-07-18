import FWCore.ParameterSet.Config as cms


def setup(process, binary_files, tree_files):
    """Pede-specific setup.
    
    Arguments:
    - `process`: cms.Process object
    - `binary_files`: list of binary files to be read by pede
    - `tree_files`: list of ROOT files created in the mille step
    """

    # enforce that alignment record is written to DB
    # --------------------------------------------------------------------------
    process.AlignmentProducer.saveToDB = True

    # setup database output module
    # --------------------------------------------------------------------------
    from CondCore.CondDB.CondDB_cfi import CondDB
    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
        CondDB,
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
    process.PoolDBOutputService.connect = "sqlite_file:alignments_MP.db"
    
    
    # Reconfigure parts of the algorithm configuration
    # --------------------------------------------------------------------------
    process.AlignmentProducer.algoConfig.mergeBinaryFiles = binary_files
    process.AlignmentProducer.algoConfig.mergeTreeFiles   = tree_files
    
    
    # Set a new source and path.
    # --------------------------------------------------------------------------
    process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
    process.source = cms.Source("EmptySource")
    process.dump = cms.EDAnalyzer("EventContentAnalyzer")
    process.p = cms.Path(process.dump)
