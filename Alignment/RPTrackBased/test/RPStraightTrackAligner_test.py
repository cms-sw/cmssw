import FWCore.ParameterSet.Config as cms

process = cms.Process("AlignmentSimulation")

# minimum of logs
process.load("Configuration.TotemCommon.LoggerMin_cfi")

# random seeds
process.load("Configuration.TotemCommon.RandomNumbers_cfi")
process.RandomNumberGeneratorService.moduleSeeds.RPFastStationSimulation = 1

# set number of events
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000000)
)

# station simulation
process.load("Configuration.TotemCommon.RandomNumbers_cfi")
process.load("Alignment.RPFastSimulation.RPFastStationSimulation_cfi")
process.RPFastStationSimulation.verbosity = 1
process.RPFastStationSimulation.pitch = 66E-3
process.RPFastStationSimulation.roundToPitch = True
process.RPFastStationSimulation.dscrWidth = 10E-3
process.RPFastStationSimulation.dscReduceUncertainty = False

process.RPFastStationSimulation.position_distribution.type = 'gauss'
process.RPFastStationSimulation.position_distribution.x_width = 6
process.RPFastStationSimulation.position_distribution.y_width = 8

process.RPFastStationSimulation.angular_distribution.type = 'gauss'
process.RPFastStationSimulation.angular_distribution.x_width = 0.1E-3
process.RPFastStationSimulation.angular_distribution.y_width = 0.1E-3

# optics parameters and ideal geometry
process.load("Configuration.TotemOpticsConfiguration.OpticsConfig_7000GeV_1535_cfi")
process.load("Geometry.VeryForwardGeometry.geometryRP_cfi")
process.XMLIdealGeometryESSource.geomXMLFiles.append('Geometry/VeryForwardData/data/' + 'RP_V:2.7_H:3.3' + '/RP_Dist_Beam_Cent.xml')

# (mis)alignments
process.load("Geometry.VeryForwardGeometryBuilder.TotemRPIncludeAlignments_cfi")
process.TotemRPIncludeAlignments.MisalignedFiles = cms.vstring()
process.TotemRPIncludeAlignments.RealFiles = cms.vstring()

# aligner
process.load("Alignment.RPTrackBased.RPStraightTrackAligner_cfi")
process.RPStraightTrackAligner.verbosity = 5
process.RPStraightTrackAligner.maxEvents = 1000

process.RPStraightTrackAligner.RPIds = [120,121,122,123,124,125]
process.RPStraightTrackAligner.z0 = 217000

process.RPStraightTrackAligner.singularLimit = 1E-10
process.RPStraightTrackAligner.useExternalFitter = True

process.RPStraightTrackAligner.minimumHitsPerProjectionPerRP = 4
process.RPStraightTrackAligner.removeImpossible = True
process.RPStraightTrackAligner.requireBothUnits = True
process.RPStraightTrackAligner.requireOverlap = False
process.RPStraightTrackAligner.requireAtLeast3PotsInOverlap = True
process.RPStraightTrackAligner.oneRotZPerPot = False
process.RPStraightTrackAligner.cutOnChiSqPerNdf = True
process.RPStraightTrackAligner.chiSqPerNdfCut = 5000
process.RPStraightTrackAligner.maxResidualToSigma = 100

process.RPStraightTrackAligner.resolveShR = True
process.RPStraightTrackAligner.resolveRotZ = True
process.RPStraightTrackAligner.resolveShZ = False
process.RPStraightTrackAligner.resolveRPShZ = False

process.RPStraightTrackAligner.algorithms = cms.vstring('Ideal', 'Jan')
process.RPStraightTrackAligner.constraintsType = "final"

process.RPStraightTrackAligner.useExtendedRotZConstraint = True
process.RPStraightTrackAligner.useZeroThetaRotZConstraint = True
process.RPStraightTrackAligner.useExtendedShZConstraints = True
process.RPStraightTrackAligner.useExtendedRPShZConstraint = True

process.RPStraightTrackAligner.homogeneousConstraints.RotZ_values = cms.vdouble(0, 0, 0, 0)

process.RPStraightTrackAligner.fixedDetectorsConstraints.ShR.ids = cms.vuint32(1200, 1201, 1248, 1249)
process.RPStraightTrackAligner.fixedDetectorsConstraints.RotZ.ids = cms.vuint32(1200, 1201)
process.RPStraightTrackAligner.fixedDetectorsConstraints.ShZ.ids = cms.vuint32(1200, 1201, 1248, 1249)
process.RPStraightTrackAligner.fixedDetectorsConstraints.RPShZ.ids = cms.vuint32(1200, 1240)
process.RPStraightTrackAligner.fixedDetectorsConstraints.RPShZ.values = cms.vdouble(0, 0)

process.RPStraightTrackAligner.JanAlignmentAlgorithm.stopOnSingularModes = False

result_dir = '.'

process.RPStraightTrackAligner.saveIntermediateResults = False
process.RPStraightTrackAligner.taskDataFileName = result_dir + "/task_data.root"

process.RPStraightTrackAligner.fileNamePrefix = result_dir + '/results_'
process.RPStraightTrackAligner.cumulativeFileNamePrefix = result_dir + '/cumulative_results_'
process.RPStraightTrackAligner.expandedFileNamePrefix = result_dir + '/cumulative_expanded_results_'
process.RPStraightTrackAligner.factoredFileNamePrefix = result_dir + '/cumulative_factored_results_'

process.RPStraightTrackAligner.buildDiagnosticPlots = False
process.RPStraightTrackAligner.JanAlignmentAlgorithm.buildDiagnosticPlots = False
process.RPStraightTrackAligner.diagnosticsFile = ''

process.p = cms.Path(process.RPFastStationSimulation * process.RPStraightTrackAligner)
