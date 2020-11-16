import FWCore.ParameterSet.Config as cms

process = cms.Process("ppsModifySingularModesTests")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# data source
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")
del(process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles[-1])
process.XMLIdealGeometryESSource_CTPPS.geomXMLFiles.append("CalibPPS/AlignmentRelative/test/test_with_mc/RP_Dist_Beam_Cent.xml")

# input
input = "CalibPPS/AlignmentRelative/test/test_modify_singular_modes/input.xml"

# initial alignments
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(input)

# worker
process.load("CalibPPS.AlignmentRelative.ppsModifySingularModes_cfi")
process.ppsModifySingularModes.inputFile = input
process.ppsModifySingularModes.outputFile = 'output.xml'

# x, y and z: mm
# rho in rad

process.ppsModifySingularModes.z1 = 213.000 * 1E3 # 56-210-fr-vr
process.ppsModifySingularModes.z2 = 220.000 * 1E3 # 56-220-fr-vr

process.ppsModifySingularModes.de_x1 = +0.100
process.ppsModifySingularModes.de_x2 = -0.100

process.ppsModifySingularModes.de_y1 = +0.200
process.ppsModifySingularModes.de_y2 = -0.200

process.ppsModifySingularModes.de_rho1 = -0.005
process.ppsModifySingularModes.de_rho2 = -0.010

# processing sequence
process.p = cms.Path(
  process.ppsModifySingularModes
)
