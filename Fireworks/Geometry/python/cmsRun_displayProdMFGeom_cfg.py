#################################################
#
# Please run the script with cmsRun:
# 
# cmsRun cmsRun_displayProdMFGeom_cfg.py
#
#################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
     geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldParameters_07.xml',
        'Geometry/CMSCommonData/data/materials.xml'),
     rootNodeName = cms.string('cms:MCMS')
)





process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )


process.source = cms.Source("EmptySource")

process.EveService = cms.Service("EveService")

### Extractor of geometry needed to display it in Eve.
### Required for "DummyEvelyser".
process.add_( cms.ESProducer(
        "TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(8)
))



process.disp = cms.EDProducer("MFProducer",
        mapDensityX = cms.untracked.uint32(10),
        mapDensityY = cms.untracked.uint32(10),
        mapDensityZ = cms.untracked.uint32(10),
        minX = cms.untracked.double(-18.0),
        maxX = cms.untracked.double(18.0),
        minY = cms.untracked.double(-18.0),
        maxY = cms.untracked.double(18.0),
        minZ = cms.untracked.double(-18.0),
        maxZ = cms.untracked.double(18.0))


process.dump = cms.EDAnalyzer(
    "DisplayGeom",
        level   = cms.untracked.int32(2),
           MF   = cms.untracked.int32(True),
    MF_plane_d1 = cms.vdouble(0, 1200., 1200.),
    MF_plane_d2 = cms.vdouble(1200, 0., 0.),
    MF_plane_N  = cms.untracked.uint32(400),
    MF_plane_draw_dir =  cms.untracked.int32(False)
)

process.p = cms.Path(process.dump+process.disp)
