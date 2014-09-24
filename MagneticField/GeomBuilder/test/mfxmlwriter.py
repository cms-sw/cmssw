import FWCore.ParameterSet.Config as cms

process = cms.Process("MagneticFieldXMLWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")


#process.load('Configuration.Geometry.MagneticFieldGeometry_cff')

GEOMETRY_VERSION = 90322
#GEOMETRY_VERSION = 120812

#FIXME: Version 90322 and 71212 are identical.
#FIXME Materials is not needed?
if (GEOMETRY_VERSION == 90322) : 
  process.XMLIdealGeometryESSource  = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms.xml',
        'Geometry/CMSCommonData/data/cmsMagneticField.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
  )
elif (GEOMETRY_VERSION == 120812) :
  process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        #FIXME: material description is for the LARGE version!!
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1_v7_large.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_2_v7_large.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
  )




process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.BigXMLWriter = cms.EDAnalyzer("OutputMagneticFieldDDToDDL",
                              rotNumSeed = cms.int32(0),
                              fileName = cms.untracked.string("./mfGeometry_"+str(GEOMETRY_VERSION)+".xml")
                              )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.BigXMLWriter)

