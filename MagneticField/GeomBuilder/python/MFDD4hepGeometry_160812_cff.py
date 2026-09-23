"""
Build the magfield geometry version 160812 from XML files using DD4hep.
"""

import FWCore.ParameterSet.Config as cms

DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath('MagneticField/GeomBuilder/data/cms-mf-geometry_160812.xml'),
    rootDDName = cms.string('cmsMagneticField:MAGF'),
    appendToDataLabel = cms.string('magfield')
    )

DDCompactViewMFESProducer = cms.ESProducer("DDCompactViewMFESProducer",
                                            appendToDataLabel = cms.string('magfield')
                                           )
