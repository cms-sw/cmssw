import FWCore.ParameterSet.Config as cms

# CRack geometry
XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
  geomXMLFiles = cms.vstring(
    'Geometry/CMSCommonData/data/materials.xml',
    'Geometry/CMSCommonData/data/rotations.xml',
    'Geometry/TrackerCommonData/data/CRack/cms.xml',
    'Geometry/TrackerCommonData/data/CRack/tobmaterial.xml',
    'Geometry/TrackerCommonData/data/tobmodpar.xml',
    'Geometry/TrackerCommonData/data/tobmodule0.xml',
    'Geometry/TrackerCommonData/data/tobmodule2.xml',
    'Geometry/TrackerCommonData/data/tobmodule4.xml',
    'Geometry/TrackerCommonData/data/tobrodpar.xml',
    'Geometry/TrackerCommonData/data/tobrod0c.xml',
    'Geometry/TrackerCommonData/data/tobrod0l.xml',
    'Geometry/TrackerCommonData/data/tobrod0h.xml',
    'Geometry/TrackerCommonData/data/tobrod1l.xml',
    'Geometry/TrackerCommonData/data/tobrod1h.xml',
    'Geometry/TrackerCommonData/data/tobrod2c.xml',
    'Geometry/TrackerCommonData/data/tobrod2l.xml',
    'Geometry/TrackerCommonData/data/tobrod2h.xml',
    'Geometry/TrackerCommonData/data/tobrod4c.xml',
    'Geometry/TrackerCommonData/data/tobrod4l.xml',
    'Geometry/TrackerCommonData/data/tobrod4h.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_DSH_L1.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_DSH_L2.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_DSL_L1.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_DSL_L2.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_SS4H.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_SS4L.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_SS6H.xml',
    'Geometry/TrackerCommonData/data/CRack/tobrod_SS6L.xml',
    'Geometry/TrackerCommonData/data/CRack/tob.xml',
    'Geometry/TrackerCommonData/data/CRack/tracker.xml',
    'Geometry/TrackerCommonData/data/CRack/trackerStructureTopology.xml',
    'Geometry/TrackerCommonData/data/CRack/trackersens_2DS_5SS6_5SS4.xml',
    'Geometry/TrackerCommonData/data/CRack/trackerRecoMaterial_2DS_5SS6_5SS4.xml',
    'Geometry/TrackerCommonData/data/CRack/trackerProdCuts_2DS_5SS6_5SS4.xml'
  ),
  rootNodeName = cms.string('cms:OCMS')
)

