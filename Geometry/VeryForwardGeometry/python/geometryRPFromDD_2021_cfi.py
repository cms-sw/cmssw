import FWCore.ParameterSet.Config as cms

# common and strip files
totemGeomXMLFiles = cms.vstring(
    'Geometry/CMSCommonData/data/materials.xml',
    'Geometry/CMSCommonData/data/rotations.xml',
    'Geometry/CMSCommonData/data/extend/cmsextent.xml',
    'Geometry/CMSCommonData/data/cms/2017/v1/cms.xml',
    'Geometry/CMSCommonData/data/beampipe/2017/v1/beampipe.xml',
    'Geometry/CMSCommonData/data/cmsBeam.xml',
    'Geometry/CMSCommonData/data/cmsMother.xml',
    'Geometry/CMSCommonData/data/mgnt.xml',
    'Geometry/TrackerCommonData/data/trackermaterial/2021/v1/trackermaterial.xml',
    'Geometry/TrackerCommonData/data/pixfwdMaterials/2021/v2/pixfwdMaterials.xml',
    'Geometry/ForwardCommonData/data/forward.xml',
    'Geometry/ForwardCommonData/data/totemRotations.xml',
    'Geometry/ForwardCommonData/data/totemMaterials.xml',
    'Geometry/VeryForwardData/data/RP_Box.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_000.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_001.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_002.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_003.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_004.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_005.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_020.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_021.xml',
#    'Geometry/VeryForwardData/data/RP_Box/RP_Box_022.xml',
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
#    'Geometry/VeryForwardData/data/RP_Box/RP_Box_122.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_123.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_124.xml',
    'Geometry/VeryForwardData/data/RP_Box/RP_Box_125.xml',
    'Geometry/VeryForwardData/data/RP_Hybrid/v1/RP_Hybrid.xml',
    'Geometry/VeryForwardData/data/RP_Materials/v1/RP_Materials.xml',
    'Geometry/VeryForwardData/data/RP_Transformations.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_000.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_001.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_002.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_004.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_005.xml',
#    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_022.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_024.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_025.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_100.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_101.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_102.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_104.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_105.xml',
#    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_122.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_124.xml',
    'Geometry/VeryForwardData/data/RP_Detectors_Assembly/RP_Detectors_Assembly_125.xml',
    'Geometry/VeryForwardData/data/RP_Device.xml',
    'Geometry/VeryForwardData/data/RP_Vertical_Device/2021/Reco/v1/RP_Vertical_Device.xml',
    'Geometry/VeryForwardData/data/RP_Horizontal_Device/2021/Reco/v1/RP_Horizontal_Device.xml',
    'Geometry/VeryForwardData/data/CTPPS_2021/Stations/Reco/v1/RP_220_Right_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_2021/Stations/Reco/v1/RP_210_Right_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_2021/Stations/Reco/v1/RP_220_Left_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_2021/Stations/Reco/v1/RP_210_Left_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_2021/Stations/Reco/v1/RP_Stations_Assembly.xml',

    'Geometry/VeryForwardData/data/RP_Sensitive_Dets.xml',
    'Geometry/VeryForwardData/data/CTPPS_2021/Cuts_Per_Region/Reco/v1/RP_Cuts_Per_Region.xml',

    'Geometry/VeryForwardData/data/CTPPS_2021/RP_Param_Beam_Region/v1/RP_Param_Beam_Region.xml'
    )

# diamond files
ctppsDiamondGeomXMLFiles = cms.vstring(
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Materials.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Transformations.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_X_Distance.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_Parameters.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Station_Parameters.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Horizontal_Pot.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Positive_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_Timing_Negative_Station.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_2021/Reco/v1/CTPPS_Timing_Stations_Assembly.xml',
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
    'Geometry/VeryForwardData/data/CTPPS_Diamond_2021/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane1.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_2021/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane2.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_2021/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane3.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_2021/CTPPS_Diamond_Planes/CTPPS_Diamond_Plane4.xml',
    'Geometry/VeryForwardData/data/CTPPS_Diamond_2021/CTPPS_Diamond_Detector_Assembly.xml'
    )


# Totem Timing files
totemTimingGeomXMLFiles = cms.vstring(
    # UFSDetectors
    'Geometry/VeryForwardData/data/TotemTiming/TotemTiming_Dist_Beam_Cent.xml',
    'Geometry/VeryForwardData/data/TotemTiming/TotemTiming_DetectorAssembly.xml',
    'Geometry/VeryForwardData/data/TotemTiming/TotemTiming_Parameters.xml',
    'Geometry/VeryForwardData/data/TotemTiming/TotemTiming_Plane.xml',
    'Geometry/VeryForwardData/data/TotemTiming/TotemTiming_Station.xml',
    )

# pixel files
ctppsPixelGeomXMLFiles = cms.vstring(
            'Geometry/VeryForwardData/data/CTPPS_Pixel_2021/Modules/v1/ppstrackerMaterials.xml',
            'Geometry/VeryForwardData/data/CTPPS_Pixel_2021/Modules/v1/PPS_Pixel_Module_2x2_Run3.xml',
            'Geometry/VeryForwardData/data/CTPPS_Pixel_2021/Modules/v1/PPS_Pixel_Sens.xml',
            'Geometry/VeryForwardData/data/CTPPS_Pixel_2021/Assembly/v1/CTPPS_Pixel_Assembly_Box_Real_003.xml',
            'Geometry/VeryForwardData/data/CTPPS_Pixel_2021/Assembly/v1/CTPPS_Pixel_Assembly_Box_Real_023.xml',
            'Geometry/VeryForwardData/data/CTPPS_Pixel_2021/Assembly/v1/CTPPS_Pixel_Assembly_Box_Real_103.xml',
            'Geometry/VeryForwardData/data/CTPPS_Pixel_2021/Assembly/v1/CTPPS_Pixel_Assembly_Box_Real_123.xml'
    )

XMLIdealGeometryESSource_CTPPS = cms.ESSource("XMLIdealGeometryESSource",
                                              geomXMLFiles = totemGeomXMLFiles + ctppsDiamondGeomXMLFiles + totemTimingGeomXMLFiles + ctppsPixelGeomXMLFiles,
                                              rootNodeName = cms.string('cms:CMSE')
                                              )

# position of RPs
XMLIdealGeometryESSource_CTPPS.geomXMLFiles.append("Geometry/VeryForwardData/data/CTPPS_2021/RP_Dist_Beam_Cent/Reco/v1/RP_Dist_Beam_Cent.xml")

ctppsGeometryESModule = cms.ESProducer("CTPPSGeometryESModule",
    verbosity = cms.untracked.uint32(1),
    isRun2 = cms.bool(False),
    compactViewTag = cms.string('XMLIdealGeometryESSource_CTPPS')
)
