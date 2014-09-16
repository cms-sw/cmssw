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


process.dump = cms.EDAnalyzer("DisplayGeom",
    level = cms.untracked.int32(2),
    MF = cms.untracked.int32(True),           #Display the MF geometry instead of detector geometry
    MF_component = cms.untracked.string("B"), #Field map: None, B, AbsBZ, AbsBR, AvsBphi, BR, Bphi
    MF_pickable = cms.untracked.bool(True),   #Field map: pickable values
    
#Field map on ZY plane @ X=0
    MF_plane_d0 = cms.untracked.vdouble(0, -900, -2000),
    MF_plane_d1 = cms.vdouble(0, -900., 2000),
    MF_plane_d2 = cms.vdouble(0, 900., -2000.),

#Field map on XY plane @ Z=0
#    MF_plane_d0 = cms.untracked.vdouble(-900, -900, 0.),
#    MF_plane_d1 = cms.vdouble(-900, 900, 0.),
#    MF_plane_d2 = cms.vdouble(900, -900, 0.),

    MF_plane_N  = cms.untracked.uint32(500), #Field map bins
   
    MF_plane_draw_dir =  cms.untracked.int32(False)
)

process.p = cms.Path(process.dump)

