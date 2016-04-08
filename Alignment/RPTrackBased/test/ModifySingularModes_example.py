import FWCore.ParameterSet.Config as cms

process = cms.Process("ModifySingularModes")

input = "Alignment/RPData/LHC/2011_10_20_1/sr+hsx/56_220.xml"
output = "./output.xml"

# minimum of logs
process.load("Configuration.TotemCommon.LoggerMin_cfi")

# empty source
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# nominal geometry
process.load("Geometry.VeryForwardGeometry.geometryRP_cfi")
process.XMLIdealGeometryESSource.geomXMLFiles.append("Geometry/VeryForwardData/data/2011_10_20_1/RP_Dist_Beam_Cent.xml")
process.TotemRPGeometryESModule = cms.ESProducer("TotemRPGeometryESModule")

# include initial alignments
process.load("Geometry.VeryForwardGeometryBuilder.TotemRPIncludeAlignments_cfi")
process.TotemRPIncludeAlignments.RealFiles = cms.vstring(input)

# add singular mode correction
process.load("Alignment.RPTrackBased.ModifySingularModes_cfi")
process.ModifySingularModes.inputFile = input
process.ModifySingularModes.outputFile = output

process.ModifySingularModes.z1 = 214628 # near unit
process.ModifySingularModes.z2 = 220000 # far unit

process.ModifySingularModes.de_x1 = -1  # 1mm shift left
process.ModifySingularModes.de_x2 = 0

process.ModifySingularModes.de_y1 = 1 # 1mm shift up
process.ModifySingularModes.de_y2 = +0

process.ModifySingularModes.de_rho1 = 0.010 # 10mrad rotation CCW
process.ModifySingularModes.de_rho2 = 0.0

process.p = cms.Path(process.ModifySingularModes)
