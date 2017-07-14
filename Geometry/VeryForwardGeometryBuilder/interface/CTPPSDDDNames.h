/****************************************************************************
*
* Authors:
*	Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_CTPPSDDDNames_H
#define Geometry_VeryForwardGeometryBuilder_CTPPSDDDNames_H


/// DDD names of sensors
const char DDD_TOTEM_RP_SENSOR_NAME[] = "RP_Silicon_Detector";
const char DDD_CTPPS_PIXELS_SENSOR_NAME[] = "RPixWafer";
//const char DDD_CTPPS_DIAMONDS_SENSOR_NAME[] = "CTPPS_Diamond_Plane";
const char DDD_CTPPS_DIAMONDS_SEGMENT_NAME[] = "CTPPS_Diamond_Segment";


/// DDD names of RP volumes
const char DDD_TOTEM_RP_RP_NAME[] = "RP_box_primary_vacuum";
const char DDD_CTPPS_PIXELS_RP_NAME[] = "RP_box_primary_vacuum"; // distiction between strip and pixel RPs is done based on copyNumbers
const char DDD_CTPPS_DIAMONDS_RP_NAME[] = "CTPPS_Diamond_Main_Box";

#endif 
