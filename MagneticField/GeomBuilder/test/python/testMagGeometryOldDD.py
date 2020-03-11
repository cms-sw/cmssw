# This cfi shows an example of how to activate some debugging tests of the field
# map geometry and data tables.
# For further information, please refer to
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMagneticField#Development_workflow
#
# Please note that the configuration shown below does not produce a complete field map.
# It should not be used as an example of setting a field map.
# For a full test of the validity of a given field map, please use:
# MagneticField/Engine/test/regression.py
# which is a superset of the test that is run here.

import FWCore.ParameterSet.Config as cms

process = cms.Process("MagneticFieldTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring( '*' ),
    destinations = cms.untracked.vstring('cout'),
    categories     = cms.untracked.vstring ( '*' ),
    cout = cms.untracked.PSet(
      noLineBreaks = cms.untracked.bool(True),
      INFO  =  cms.untracked.PSet (limit = cms.untracked.int32(-1)),
      DEBUG =  cms.untracked.PSet (limit = cms.untracked.int32(-1)),
      WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
      ),
      ERROR = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
      ),
      threshold = cms.untracked.string('DEBUG'),
      default =  cms.untracked.PSet (limit = cms.untracked.int32(-1))
    )
)

process.magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_1.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_2.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
    rootNodeName = cms.string('cmsMagneticField:MAGF')
)

# avoid interference with EmptyESSource in uniformMagneticField.cfi
process.es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource","magfield")


process.MagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
                                              DDDetector = cms.ESInputTag('', 'magfield'),
                                              appendToDataLabel = cms.string(''),
                                              useParametrizedTrackerField = cms.bool(False),
                                              label = cms.untracked.string(''),
                                              attribute = cms.string('magfield'),
                                              value = cms.string('magfield'),
                                              paramLabel = cms.string(''),
                                              version = cms.string('fake'),
                                              geometryVersion = cms.int32(160812),
                                              debugBuilder = cms.untracked.bool(False), # Set to True to activate full debug
                                              cacheLastVolume = cms.untracked.bool(True),
                                              scalingVolumes = cms.vint32(),
                                              scalingFactors = cms.vdouble(),

                                              gridFiles = cms.VPSet()
                                               )

process.test = cms.EDAnalyzer("testMagGeometryAnalyzer",
                              DDDetector = cms.ESInputTag('', 'magfield')
                              )

process.p = cms.Path(process.test)
