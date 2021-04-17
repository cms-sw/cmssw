import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process("trackBasedAlignment", eras.Run2_2018)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# input data
process.source = cms.Source("PoolSource",
    skipBadFiles = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
$inputFiles
    ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange($lsList),
    inputCommands = cms.untracked.vstring(
      "drop *",
      "keep TotemRPRecHitedmDetSetVector_*_*_*",
      "keep CTPPSPixelRecHitedmDetSetVector_*_*_*",
    )
)

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")
del(process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles[-1])
process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles.append("$geometry/RP_Dist_Beam_Cent.xml")

# initial alignments
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring($alignmentFiles)

# reco modules
process.load("RecoPPS.Local.totemRPLocalReconstruction_cff")

process.load("RecoPPS.Local.ctppsPixelLocalReconstruction_cff")

process.load("RecoPPS.Local.ctppsLocalTrackLiteProducer_cff")
process.ctppsLocalTrackLiteProducer.includeDiamonds = False

# aligner
process.load("CalibPPS.AlignmentRelative.ppsStraightTrackAligner_cfi")

process.ppsStraightTrackAligner.verbosity = 1

process.ppsStraightTrackAligner.tagUVPatternsStrip = cms.InputTag("totemRPUVPatternFinder")
process.ppsStraightTrackAligner.tagDiamondHits = cms.InputTag("")
process.ppsStraightTrackAligner.tagPixelHits = cms.InputTag("")
process.ppsStraightTrackAligner.tagPixelLocalTracks = cms.InputTag("ctppsPixelLocalTracks")

process.ppsStraightTrackAligner.maxEvents = int($maxEvents)

process.ppsStraightTrackAligner.rpIds = [$rps]
process.ppsStraightTrackAligner.excludePlanes = cms.vuint32($excludePlanes)
process.ppsStraightTrackAligner.z0 = $z0

process.ppsStraightTrackAligner.maxResidualToSigma = $maxResidualToSigma
process.ppsStraightTrackAligner.minimumHitsPerProjectionPerRP = $minimumHitsPerProjectionPerRP

process.ppsStraightTrackAligner.removeImpossible = True
process.ppsStraightTrackAligner.requireNumberOfUnits = $requireNumberOfUnits
process.ppsStraightTrackAligner.requireOverlap = $requireOverlap
process.ppsStraightTrackAligner.requireAtLeast3PotsInOverlap = $requireAtLeast3PotsInOverlap
process.ppsStraightTrackAligner.additionalAcceptedRPSets = "$additionalAcceptedRPSets"

process.ppsStraightTrackAligner.cutOnChiSqPerNdf = True
process.ppsStraightTrackAligner.chiSqPerNdfCut = $chiSqPerNdfCut

process.ppsStraightTrackAligner.maxTrackAx = $maxTrackAx
process.ppsStraightTrackAligner.maxTrackAy = $maxTrackAy

optimize="$optimize"
process.ppsStraightTrackAligner.resolveShR = $resolveShR
process.ppsStraightTrackAligner.resolveShZ = False
process.ppsStraightTrackAligner.resolveRotZ = $resolveRotZ

process.ppsStraightTrackAligner.constraintsType = "standard"
process.ppsStraightTrackAligner.standardConstraints.units = cms.vuint32($final_constraints_units)
process.ppsStraightTrackAligner.oneRotZPerPot = $oneRotZPerPot
process.ppsStraightTrackAligner.useEqualMeanUMeanVRotZConstraints = $useEqualMeanUMeanVRotZConstraints

process.ppsStraightTrackAligner.algorithms = cms.vstring("Jan")

process.ppsStraightTrackAligner.JanAlignmentAlgorithm.stopOnSingularModes = False

results_dir="$results_dir"

process.ppsStraightTrackAligner.taskDataFileName = "" # results_dir + "/task_data.root"

process.ppsStraightTrackAligner.fileNamePrefix = results_dir + "/results_iteration_"
process.ppsStraightTrackAligner.expandedFileNamePrefix = results_dir + "/results_cumulative_expanded_"
process.ppsStraightTrackAligner.factoredFileNamePrefix = results_dir + "/results_cumulative_factored_"

process.ppsStraightTrackAligner.diagnosticsFile = results_dir + '/diagnostics.root'
process.ppsStraightTrackAligner.buildDiagnosticPlots = $buildDiagnosticPlots
process.ppsStraightTrackAligner.JanAlignmentAlgorithm.buildDiagnosticPlots = $buildDiagnosticPlots

# processing sequence
process.p = cms.Path(
  # it is important to re-run part of the reconstruction as it may influence
  # the choice of rec-hits used in the alignment
  process.totemRPUVPatternFinder
  * process.totemRPLocalTrackFitter
  * process.ctppsPixelLocalTracks
  * process.ctppsLocalTrackLiteProducer

  * process.ppsStraightTrackAligner
)
