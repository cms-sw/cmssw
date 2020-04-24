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
//const std::string DDD_CTPPS_DIAMONDS_SENSOR_NAME = "CTPPS_Diamond_Plane";
const std::string DDD_CTPPS_DIAMONDS_SEGMENT_NAME = "CTPPS_Diamond_Segment";
const std::string DDD_CTPPS_UFSD_SEGMENT_NAME = "CTPPS_UFSD_Segment";


/// DDD names of RP volumes
const std::string DDD_TOTEM_RP_RP_NAME = "RP_box_primary_vacuum";
const std::string DDD_CTPPS_PIXELS_RP_NAME = "RP_box_primary_vacuum"; // distiction between strip and pixel RPs is done based on copyNumbers
const std::string DDD_CTPPS_DIAMONDS_RP_NAME = "CTPPS_Diamond_Main_Box";
const std::string DDD_CTPPS_UFSD_PLANE_NAME = "CTPPS_UFSD_Plane";

#endif 
