#################################################
#
# Please run the script with cmsRun:
# 
# cmsRun displayMFGeom_cfg.py
#
#################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

#process.load("Configuration.StandardSequences.GeometryExtended_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")

MFGeom=True;

if MFGeom: 
 process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
      geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
         'Geometry/CMSCommonData/data/cms.xml', 
         'Geometry/CMSCommonData/data/cmsMagneticField.xml',
#         'MagneticField/GeomBuilder/data/test.xml',
          # for 090322 and older
#         'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',


          # for 160812
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_1.xml',
        'MagneticField/GeomBuilder/data/MagneticFieldVolumes_160812_2.xml',

        'Geometry/CMSCommonData/data/materials.xml'),
      rootNodeName = cms.string('cmsMagneticField:MAGF')
 ) 
else : 
 process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")



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

#Z_SECTION=-165  #S3 chimney
#Z_SECTION=165   #S4 chimney
#Z_SECTION=760  #YE1
#Z_SECTION=880  #YE2
#Z_SECTION=979  #YE3
#Z_SECTION=1069 #YE4 edge
#Z_SECTION=1072 #YE4
#Z_SECTION=-1519.4957 #CASTOR

#Z_SECTION=1434 #?
#Z_SECTION=1548 #?




process.dump = cms.EDAnalyzer("DisplayGeom",                              
#    process.dump.level   = cms.untracked.int32(2),
    MF   = cms.untracked.int32(MFGeom),            #Display the MF geometry instead of detector geometry
    MF_component = cms.untracked.string("B"), #Field map: None, B, AbsBZ, AbsBR, AvsBphi, BR, Bphi
    MF_pickable = cms.untracked.bool(False),     #Field map: pickable values
    

#Field map on ZY plane @ X=0 (Top-right quarter)
#    MF_plane_d0 = cms.untracked.vdouble(0, 0, 0),
#    MF_plane_d1 = cms.vdouble(0, 0., 2000),
#    MF_plane_d2 = cms.vdouble(0, 900., 0.),

#Field map on ZY plane @ X=0 (full CMS)
    MF_plane_d0 = cms.untracked.vdouble(0, -900, -2400),
    MF_plane_d1 = cms.vdouble(0, -900., 2400),
    MF_plane_d2 = cms.vdouble(0, 900., -2400.),

#Field map on XY plane @ Z=0
#    MF_plane_d0 = cms.untracked.vdouble(-900, -900, 0.),
#    MF_plane_d1 = cms.vdouble(-900, 900, 0.),
#    MF_plane_d2 = cms.vdouble(900, -900, 0.),

#Field map on XY plane @ S3chimney
#    MF_plane_d0 = cms.untracked.vdouble(-900, -900, -165.),
#    MF_plane_d1 = cms.vdouble(-900, 900, -165.),
#    MF_plane_d2 = cms.vdouble(900, -900, -165.),

#Field map on XY plane @ S11 feet
#    MF_plane_d0 = cms.untracked.vdouble(0, -900, 0),
#    MF_plane_d1 = cms.vdouble(0, 0, 0),
#    MF_plane_d2 = cms.vdouble(900, -900, 0),

#XY plane @ Z_SECTION
#    MF_plane_d0 = cms.untracked.vdouble(-900, -900, Z_SECTION),
#    MF_plane_d1 = cms.vdouble(-900, 900, Z_SECTION),
#    MF_plane_d2 = cms.vdouble(900, -900, Z_SECTION),

#CASTOR detail
#    MF_plane_d0 = cms.untracked.vdouble(-200, -200, -1519.49571429),
#    MF_plane_d1 = cms.vdouble(-200, 200, -1519.49571429),
#    MF_plane_d2 = cms.vdouble(200, -200, -1519.49571429),


    MF_plane_N  = cms.untracked.uint32(1000), #Field map bins
#    MF_plane_N2  = cms.untracked.uint32(500),
   
    MF_plane_draw_dir =  cms.untracked.int32(False)
)

process.p = cms.Path(process.dump)

