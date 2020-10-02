import FWCore.ParameterSet.Config as cms

process = cms.Process("ppsTrackBasedAlignmentTest")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# random seeds
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
  ppsFastLocalSimulation = cms.PSet(
    initialSeed = cms.untracked.uint32(81)
  )
)

# data source
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")
del(process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles[-1])
process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles.append("CalibPPS/AlignmentRelative/test/test_with_mc//RP_Dist_Beam_Cent.xml")

# initial alignments
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.MisalignedFiles = cms.vstring("CalibPPS/AlignmentRelative/test/test_with_mc/mis_alignment_sr.xml")

# simulation
process.load("CalibPPS.AlignmentRelative.ppsFastLocalSimulation_cfi")
process.ppsFastLocalSimulation.verbosity = 0
process.ppsFastLocalSimulation.z0 = 215000
process.ppsFastLocalSimulation.RPs = cms.vuint32(103, 104, 105, 123, 124, 125)
process.ppsFastLocalSimulation.roundToPitch = True

# strips: pattern recognition
process.load("RecoPPS.Local.totemRPUVPatternFinder_cfi")
process.totemRPUVPatternFinder.tagRecHit = cms.InputTag("ppsFastLocalSimulation")

# aligner
process.load("CalibPPS.AlignmentRelative.ppsStraightTrackAligner_cfi")

process.ppsStraightTrackAligner.verbosity = 1

process.ppsStraightTrackAligner.tagUVPatternsStrip = cms.InputTag("totemRPUVPatternFinder")
process.ppsStraightTrackAligner.tagDiamondHits = cms.InputTag("")
process.ppsStraightTrackAligner.tagPixelHits = cms.InputTag("ppsFastLocalSimulation")

process.ppsStraightTrackAligner.maxEvents = 1000

process.ppsStraightTrackAligner.rpIds = [103, 104, 105, 123, 124, 125]
process.ppsStraightTrackAligner.excludePlanes = cms.vuint32()
process.ppsStraightTrackAligner.z0 = process.ppsFastLocalSimulation.z0

process.ppsStraightTrackAligner.maxResidualToSigma = 100
process.ppsStraightTrackAligner.minimumHitsPerProjectionPerRP = 3

process.ppsStraightTrackAligner.removeImpossible = True
process.ppsStraightTrackAligner.requireNumberOfUnits = 2
process.ppsStraightTrackAligner.requireOverlap = False
process.ppsStraightTrackAligner.requireAtLeast3PotsInOverlap = True
process.ppsStraightTrackAligner.additionalAcceptedRPSets = ""

process.ppsStraightTrackAligner.cutOnChiSqPerNdf = True
process.ppsStraightTrackAligner.chiSqPerNdfCut = 5000

process.ppsStraightTrackAligner.maxTrackAx = 1
process.ppsStraightTrackAligner.maxTrackAy = 1

process.ppsStraightTrackAligner.resolveShR = True
process.ppsStraightTrackAligner.resolveShZ = False
process.ppsStraightTrackAligner.resolveRotZ = True

process.ppsStraightTrackAligner.constraintsType = cms.string("standard")
process.ppsStraightTrackAligner.standardConstraints.units = cms.vuint32(101, 121)
process.ppsStraightTrackAligner.useEqualMeanUMeanVRotZConstraints = False
process.ppsStraightTrackAligner.oneRotZPerPot = True

process.ppsStraightTrackAligner.algorithms = cms.vstring("Ideal", "Jan")

process.ppsStraightTrackAligner.JanAlignmentAlgorithm.stopOnSingularModes = False

process.ppsStraightTrackAligner.taskDataFileName = "" # results_dir + "/task_data.root"

process.ppsStraightTrackAligner.fileNamePrefix = "results_"
process.ppsStraightTrackAligner.expandedFileNamePrefix = "results_cumulative_expanded_"
process.ppsStraightTrackAligner.factoredFileNamePrefix = "results_cumulative_factored_"

process.ppsStraightTrackAligner.diagnosticsFile = 'diagnostics.root'
process.ppsStraightTrackAligner.buildDiagnosticPlots = True
process.ppsStraightTrackAligner.JanAlignmentAlgorithm.buildDiagnosticPlots = True

# processing sequence
process.p = cms.Path(
  process.ppsFastLocalSimulation
  * process.totemRPUVPatternFinder
  * process.ppsStraightTrackAligner
)
