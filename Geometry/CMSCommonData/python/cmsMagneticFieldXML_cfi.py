# The following comments couldn't be translated into the new config version:

#   FileInPath GeometryConfiguration ="Geometry/CMSCommonData/data/FieldConfiguration.xml"

import FWCore.ParameterSet.Config as cms

magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 'Geometry/CMSCommonData/data/cms.xml', 'Geometry/CMSCommonData/data/cmsMagneticField.xml', 'Geometry/CMSCommonData/data/MagneticFieldVolumes.xml'),
    rootNodeName = cms.string('MagneticFieldVolumes:MAGF')
)


