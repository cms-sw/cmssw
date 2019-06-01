#ifndef CASTORGEOMETRYDATA_H
#define CASTORGEOMETRYDATA_H 1

/** the ZSection position gives the absolute-value  z  limits for each section (EM, HAD)
            |||{ Al }/////////////   
            ^EM part^ HAD part   ^  
            |       |     
            |       z-had 
	   z-em

the SectorBoundaries define the sector for each section
      13,14 \15,16 |1,2 /3,4
	     -          - 
       11,12 /9,10 |7,8 \5,6

 theXChannelBoundaries positions are the x low limits for each channel (EM section)
 theZHadChannelBoundaries are the z low limits for each HAD channel at y = 0 position
 the coordinates are given with respect to the center of the enclosing volume
 as defined in the geometry: X0,Z0,Y0
 tiltangle is the 45 deg angle of the plates
 dXEMPlate, dXHADPlate is the half width of EM, HAD plates
 dYEMPlate, dyHADPlate  is half height (DY/2) of the EM, HAD sections
 dXAll is the half width (DX/2) of all sections	
 all dimmensions are in mm and rads

NOTE not final

 Panos Katsas, September 2007.

**/
static const double X0 = 0.;
static const double Z0 = 14385.0;
static const double Y0 = 0.;
static const double dYEMPlate = 62.5;
static const double dYHADPlate = 320.0;
static const double dXEMPlate = 48.0;
static const double dXHADPlate = 48.0;

static const double tiltangle = 0.7854;  // 45 degrees
static const double theZSectionBoundaries[] = {14385., 14488.};
static const double theXChannelBoundaries[] = {-48.0, -28.8, -9.6, 9.6, 28.8};
static const double theZHadChannelBoundaries[] = {
    -257.4,
    118.2,
    21.0,
    160.2,
};
static const double theHadmodulesBoundaries[] = {
    14488., 14589., 14690., 14791., 14892., 14993., 15094., 15195., 15296., 15397., 15498., 15599.};
static const double theSectorBoundaries[] = {0.,
                                             0.21817,
                                             0.4363,
                                             0.6545,
                                             0.87267,
                                             1.0908,
                                             1.309,
                                             1.52718,
                                             1.74535,
                                             1.9635,
                                             2.1817,
                                             2.39986,
                                             2.618,
                                             2.8362,
                                             3.05437,
                                             3.27254};
#endif
