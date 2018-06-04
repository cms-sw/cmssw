import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
                                        geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
                                                                   'Geometry/CMSCommonData/data/cms.xml', 
                                                                   'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
                                                                   'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_1.xml',
                                                                   'Geometry/CMSCommonData/data/materials.xml'),
                                        rootNodeName = cms.string('cms:MCMS')
                                        )
