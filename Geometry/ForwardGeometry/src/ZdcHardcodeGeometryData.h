#ifndef Geometry_ForwardGeometry_ZdcHardcodeGeometryData_H
#define Geometry_ForwardGeometry_ZdcHardcodeGeometryData_H 1

/** the ZSection position gives the absolute-value  z (low) limits for each section
            ||||||||||{  }////////   
            ^         ^   ^  
            |         |   |     
            |         |  zlow - had 
	   zlow-em   zlow-lum       

 theXChannelBoundaries positions are the x low limits for each channel (EM section)
 theZLUMChannelBoundaries are the z low limits for each channel (LUM section)
 theZHadChannelBoundaries are the z low limits for each HAD channel at y = 0 position
 this coordinates are given with respect to the center of the enclosing volume (ZDC)
 as defiened in the geometry: X0,Z0,Y0
 tiltangle is the angle of the HAD section
 YLUM is the Y position of LUM 
 dYPlate  is half height (DY/2) of the HAD and LUM sections
 dYLum is the half height (DY/2) of the LUM section
 dXPlate is the half width (DX/2) of all sections	
 all dimmensions are in mm and rads
 Edmundo Garcia, August 2007.
**/
static const double X0 = 0.;
static const double Z0 = 140000.0;
static const double Y0 = 0.;
static const double YLUM = 253.6;
static const double YRPD = 253.6;
static const double dYPlate = 62.5;
static const double dYLUM = 320.0;
static const double dYRPD = 320.0;
static const double dXPlate = 48.0;
static const double tiltangle = 0.7854;  // 45 degrees
static const double theZSectionBoundaries[] = {-500.0, -395.55, -290.0};
static const double theXChannelBoundaries[] = {-48.0, -28.8, -9.6, 9.6, 28.8};
static const double theZLUMChannelBoundaries[] = {-395.55, -346.525, -301.5};
static const double theZRPDChannelBoundaries[] = {-395.55, -346.525, -301.5};
static const double theZHadChannelBoundaries[] = {
    -257.4,
    -118.2,
    21.0,
    160.2,
};
#endif
