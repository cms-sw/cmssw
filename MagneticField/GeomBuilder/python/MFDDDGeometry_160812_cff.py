"""
Build the magfield geometry version 160812 from XML files using DDD (deprecated).
"""

import FWCore.ParameterSet.Config as cms
# Note that XMLIdealGeometryESSource puts the geometry in IdealMagneticFieldRecord or IdealGeometryRecord depending on the specified rootNodeName.
MFDDDGeometry = cms.ESSource("XMLIdealGeometryESSource",
                             geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
                                                        'Geometry/CMSCommonData/data/cms.xml', 
                                                        'Geometry/CMSCommonData/data/cmsMagneticField.xml',
                                                        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_1.xml',
                                                        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_2.xml',
                                                        'Geometry/CMSCommonData/data/materials.xml'),
                             rootNodeName = cms.string('cmsMagneticField:MAGF') 
                             )
