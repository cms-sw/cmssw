import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
                                        geomXMLFiles = cms.vstring
 ('Geometry/CMSCommonData/data/materials.xml',
  'Geometry/CMSCommonData/data/rotations.xml',
  'Geometry/TrackerCommonData/data/pixbarmaterial.xml', 
  'Geometry/TrackerCommonData/data/pixbarladder.xml', 
  'Geometry/TrackerCommonData/data/pixbarladderfull.xml', 
  'Geometry/TrackerCommonData/data/pixbarladderhalf.xml', 
  'Geometry/TrackerCommonData/data/pixbarlayer.xml', 
  'Geometry/TrackerCommonData/data/pixbarlayer0.xml', 
  'Geometry/TrackerCommonData/data/pixbarlayer1.xml', 
  'Geometry/TrackerCommonData/data/pixbarlayer2.xml', 
  'Geometry/TrackerCommonData/data/pixbar.xml', 
  'Geometry/TrackerCommonData/data/trackerpixbar.xml', 
  'Geometry/TrackerCommonData/data/tracker.xml',
  'Geometry/TrackerCommonData/data/trackermaterial.xml',
  'Geometry/TrackerCommonData/data/pixfwdMaterials.xml',
  'Geometry/CMSCommonData/data/cmsMother.xml',
  'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
  'Geometry/CMSCommonData/data/cms.xml'),
 rootNodeName = cms.string('cms:OCMS')
 )


