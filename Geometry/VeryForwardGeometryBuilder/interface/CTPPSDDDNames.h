/****************************************************************************
*
* Authors:
*	Jan Kaspar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_CTPPSDDDNames_H
#define Geometry_VeryForwardGeometryBuilder_CTPPSDDDNames_H

#include <string>

/// DDD names of sensors
const std::string DDD_TOTEM_RP_SENSOR_NAME = "RP_Silicon_Detector";
const std::string DDD_CTPPS_PIXELS_SENSOR_NAME = "RPixWafer";
const std::string DDD_CTPPS_PIXELS_SENSOR_TYPE_2x2 = "2x2";
const std::string DDD_CTPPS_DIAMONDS_SEGMENT_NAME = "CTPPS_Diamond_Segment";
const std::string DDD_CTPPS_UFSD_SEGMENT_NAME = "CTPPS_UFSD_Segment";
const std::string DDD_TOTEM_TIMING_SENSOR_TMPL = "UFSD_ch(\\d+)";

/// DDD names of RP volumes
const std::string DDD_TOTEM_RP_RP_NAME = "RP_box_primary_vacuum";
const std::string DDD_CTPPS_PIXELS_RP_NAME =
    "RP_box_primary_vacuum";  // distiction between strip and pixel RPs is done based on copyNumbers
const std::string DDD_CTPPS_DIAMONDS_RP_NAME = "CTPPS_Diamond_Main_Box";
const std::string DDD_TOTEM_TIMING_RP_NAME = "TotemTiming_Main_Box";

#endif
