import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
      'Geometry/TrackerCommonData/data/tibmaterial.xml',
      'Geometry/TrackerCommonData/data/tibmodpar.xml',
      'Geometry/TrackerCommonData/data/tibmodule0.xml',
      'Geometry/TrackerCommonData/data/tibmodule0a.xml',
      'Geometry/TrackerCommonData/data/tibmodule0b.xml',
      'Geometry/TrackerCommonData/data/tibmodule2.xml',
      'Geometry/MTCCTrackerCommonData/data/tibemptymodule2_mtcc.xml',
      'Geometry/TrackerCommonData/data/tibstringpar.xml',
      'Geometry/TrackerCommonData/data/tibstringds.xml',
      'Geometry/TrackerCommonData/data/tibstring1c.xml',
      'Geometry/TrackerCommonData/data/tibstring1lr.xml',
      'Geometry/TrackerCommonData/data/tibstring1ur.xml',
      'Geometry/MTCCTrackerCommonData/data/tibstring1_mtcc.xml',
      'Geometry/TrackerCommonData/data/tibstringss.xml',
      'Geometry/TrackerCommonData/data/tibstring2c.xml',
      'Geometry/TrackerCommonData/data/tibstring2lr.xml',
      'Geometry/TrackerCommonData/data/tibstring2ur.xml',
      'Geometry/MTCCTrackerCommonData/data/tibstring2_mtcc.xml',
      'Geometry/MTCCTrackerCommonData/data/tibemptystringss_mtcc.xml',
      'Geometry/MTCCTrackerCommonData/data/tibemptystring2lr_mtcc.xml',
      'Geometry/MTCCTrackerCommonData/data/tibemptystring2ur_mtcc.xml',
      'Geometry/MTCCTrackerCommonData/data/tibemptystring2_mtcc.xml',
      'Geometry/TrackerCommonData/data/tiblayerpar.xml',
      'Geometry/MTCCTrackerCommonData/data/tiblayer1_mtcc.xml',
      'Geometry/MTCCTrackerCommonData/data/tiblayer2_mtcc.xml',
      'Geometry/MTCCTrackerCommonData/data/tib_mtcc.xml'),
    rootNodeName = cms.string('tib_mtcc:TIB')
)


