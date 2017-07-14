import FWCore.ParameterSet.Config as cms

#ideal geometry
#XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
totemGeomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                                'Geometry/CMSCommonData/data/rotations.xml', 
                                'Geometry/CMSCommonData/data/extend/cmsextent.xml',
                                'Geometry/CMSCommonData/data/cms/2017/v1/cms.xml',
                                'Geometry/CMSCommonData/data/beampipe/2017/v1/beampipe.xml', 
                                'Geometry/CMSCommonData/data/cmsBeam.xml', 
                                'Geometry/CMSCommonData/data/cmsMother.xml', 
                                'Geometry/CMSCommonData/data/mgnt.xml', 
                                'Geometry/ForwardCommonData/data/forward.xml', 
                                'Geometry/ForwardCommonData/data/totemRotations.xml', 
                                'Geometry/ForwardCommonData/data/totemMaterials.xml', 
                                'Geometry/ForwardCommonData/data/totemt1.xml', 
                                'Geometry/ForwardCommonData/data/totemt2.xml', 
                                'Geometry/ForwardCommonData/data/ionpump.xml', 
                                'Geometry/VeryForwardData/data/RP_Box.xml',#totem->
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_000.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_001.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_002.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_003.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_004.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_005.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_020.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_021.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_022.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_023.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_024.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_025.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_100.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_101.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_102.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_103.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_104.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_105.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_120.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_121.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_122.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_123.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_124.xml',
                                'Geometry/VeryForwardData/data/RP_Box/RP_Box_125.xml',#totem<-
                                'Geometry/VeryForwardData/data/RP_Hybrid.xml',
                                'Geometry/VeryForwardData/data/RP_Materials.xml',
                                'Geometry/VeryForwardData/data/RP_Transformations.xml',
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly.xml',
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_000.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_001.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_002.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_003.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_004.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_005.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_020.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_021.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_022.xml',#totem
                                
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_024.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_025.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_100.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_101.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_102.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_103.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_104.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_105.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_120.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_121.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_122.xml',#totem
                                
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_124.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_125.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Device.xml',
                                'Geometry/VeryForwardData/data/RP_Vertical_Device.xml',#totem
                                'Geometry/VeryForwardData/data/RP_Horizontal_Device.xml',
                                'Geometry/VeryForwardData/data/RP_220_Right_Station.xml',#totem, from RP -> CTPPS(trackers and timing)
                                'Geometry/VeryForwardData/data/RP_220_Left_Station.xml',#totem, from RP -> CTPPS (trackers and timing)
                                'Geometry/VeryForwardData/data/RP_147_Right_Station.xml',#totem, from RP -> CTPPS
                                'Geometry/VeryForwardData/data/RP_147_Left_Station.xml',#totem, from RP -> CTPPS
                                'Geometry/VeryForwardData/data/RP_Stations_Assembly.xml',#totem, from RP -> CTPPS
                                'Geometry/VeryForwardData/data/RP_Sensitive_Dets.xml',
                                'Geometry/VeryForwardData/data/RP_Cuts_Per_Region.xml',#totem, from RP -> CTPPS
                                'Geometry/VeryForwardData/data/RP_Param_Beam_Region.xml')


ctppsDiamondGeomXMLFiles = cms.vstring(
    # diamond detectors
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Materials.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Transformations.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_X_Distance.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Parameters.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Station_Parameters.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Horizontal_Pot.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Positive_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Negative_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Stations_Assembly.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern1_Segment1.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern2_Segment1.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern2_Segment2.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern3_Segment1.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern3_Segment2.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern3_Segment3.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern3_Segment4.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern4_Segment1.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern4_Segment2.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern4_Segment3.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern4_Segment4.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Segments/CTPPS_Diamond_Pattern4_Segment5.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane1.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane2.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane3.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane4.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Detector_Assembly.xml'
    )

ctppsPixelGeomXMLFiles = cms.vstring(

    'Geometry/VeryForwardData/data/ppstrackerMaterials.xml',
    'Geometry/VeryForwardData/data/CTPPS_Pixel_Module.xml',
    'Geometry/VeryForwardData/data/CTPPS_Pixel_Module_2x2.xml',
    'Geometry/VeryForwardData/data/CTPPS_Pixel_Assembly_Box_Real_023.xml',
    'Geometry/VeryForwardData/data/CTPPS_Pixel_Assembly_Box_Real_123.xml',
    'Geometry/VeryForwardData/data/CTPPS_Pixel_Sens.xml'
    )


XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
                                              geomXMLFiles = totemGeomXMLFiles+ctppsDiamondGeomXMLFiles
+ctppsPixelGeomXMLFiles
,
                                              rootNodeName = cms.string('cms:CMSE')
                                              )


#XMLIdealGeometryESSource.geomXMLFiles.append("Geometry/VeryForwardData/data/RP_Position/RP_Dist_Beam_Cent.xml")
XMLIdealGeometryESSource.geomXMLFiles.append("Geometry/VeryForwardData/data/2016_ctpps_15sigma_margin0/RP_Dist_Beam_Cent.xml")

# real geometry
ctppsGeometryESModule = cms.ESProducer("CTPPSGeometryESModule",
                                         verbosity = cms.untracked.uint32(3)
                                         )

