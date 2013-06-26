import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/TrackerCommonData/data/trackermaterial.xml', 
        'Geometry/TrackerCommonData/data/tecmaterial.xml', 
        'Geometry/TrackerCommonData/data/trackerbulkhead.xml', 
        'Geometry/TrackerCommonData/data/trackerother.xml', 
        'Geometry/TrackerCommonData/data/tracker.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/TrackerCommonData/data/cms.xml'),
    rootNodeName = cms.string('cms:CMSE')
)


