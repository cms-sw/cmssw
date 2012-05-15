import FWCore.ParameterSet.Config as cms

from FWCore.MessageLogger.MessageLogger_cfi import *
process = cms.Process("MF")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
     geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldParameters_07.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
     rootNodeName = cms.string('cms:MCMS')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.source = cms.Source("EmptySource")

process.prod = cms.EDProducer("GeometryProducer",
                              MagneticField = cms.PSet(delta = cms.double(1.0)),
                              UseMagneticField = cms.bool(False),
                              UseSensitiveDetectors = cms.bool(False)
                              )

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))

process.disp = cms.EDProducer("MFProducer",
        mapDensityX = cms.untracked.uint32(10),
        mapDensityY = cms.untracked.uint32(10),
        mapDensityZ = cms.untracked.uint32(10),
        minX = cms.untracked.double(-18.0),
        maxX = cms.untracked.double(18.0),
        minY = cms.untracked.double(-18.0),
        maxY = cms.untracked.double(18.0),
        minZ = cms.untracked.double(-18.0),
        maxZ = cms.untracked.double(18.0))

process.p = cms.Path(process.prod+process.disp)
