#ifndef Point5MaterialMap_cc
#define Point5MaterialMap_cc

#include <iostream>
#include <cmath>

#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"



inline int inPlug(double vx, double vy, double vz,
		  double PlugVx = PlugOnShaftVx, double PlugVz = PlugOnShaftVz) {
  if (vy > SurfaceOfEarth && vy < SurfaceOfEarth + PlugWidth) {
    if (vx > PlugVx - PlugXlength/2. && vx < PlugVx + PlugXlength/2. &&
	vz > PlugVz - PlugZlength/2. && vz < PlugVz + PlugZlength/2.) return Plug;
    if (vz >= PlugVz - PlugZlength/2. - PlugNoseZlength && vz < PlugVz - PlugZlength/2. &&
	vx > PlugVx - PlugNoseXlength/2. && vx < PlugVx + PlugNoseXlength/2.) return Plug;
  }
  return Unknown;
}


inline int inAirAfterPlug(double vx, double vy, double vz) {
  // particles above surface of earth
  if (vy >= SurfaceOfEarth) return Air;
  
  // CMS cavern (UXC 55)
  if (std::fabs(vz) < 26548. && sqrt((vx*1.1576)*(vx*1.1576) + vy*vy) < 15460. &&
      vy > -8762) return Air;
  
  // access shaft (PX 56)
  if (vy > 0. && vy < (SurfaceOfEarth-2250.) && 
      sqrt(vx*vx + (vz-Z_PX56)*(vz-Z_PX56)) < 10250.) return Air;
  
  //surface hall ground floor
  if (vy >= SurfaceOfEarth-2250. && vy < SurfaceOfEarth) {
    if (sqrt(vx*vx + (vz-Z_PX56)*(vz-Z_PX56)) < 10250.
	&& vz-Z_PX56 > -7000. && vz-Z_PX56 < 7000.) return Air;
    if (vx > -2400. && vx < 2400. && vz-Z_PX56 >= -9800. && vz-Z_PX56 < -7000. )
      return Air;
  }
  
    // Shaft (PM 54)
  if (vy > 3233. && vy < (SurfaceOfEarth) && 
      sqrt((vx-26600.)*(vx-26600.) + (vz-30100.-Z_PX56)*(vz-30100.-Z_PX56)) < 6050.)
    //sqrt((vx-5000.)*(vx-5000.) + (vz-30100.-Z_PX56)*(vz-30100.-Z_PX56)) < 10050.)//@@@@@
    return Air;
  
  
  // Shaft (PM 56)
  if (vy > -3700. && vy < (SurfaceOfEarth) &&
      sqrt((vx-18220.)*(vx-18220.) + (vz+24227.-Z_PX56)*(vz+24227.-Z_PX56)) < 3550.)
      return Air;
  
  //Service cavern USC 55
  if (vz > -22050. && vz < 62050. &&
      sqrt((vx-29550.)*(vx-29550.) + (vy-3233.)*(vy-3233.)) < 9050. && 
      vy > -3650.) return Air;
  
  //UXC55 cavern endcap beam openings
  if (((vz >= -29278. && vz < -26548.) || (vz >= 26548 && vz <= 29278.)) &&
      sqrt(vy*vy + vx*vx) < 800.) return Air; 
  
  //Pillar access between CMS collision cavern and service cavern
  if (vx > 10000. && vx < 20500.+3050. && //TX54 galerie
      vz > 14460. && vz < 16260. &&//estimated wall thickness 1m
      vy > -8680. && vy < -1507.) return Air;
  
  if (vx > 10000. && vx < 16500. && //TX54 galerie 30.9 degree opening to UXC55
      vy > -8680. && vy < -1507. && //@vx=13300 delta(vz)=1865., 2vx=1650. delta(vz)=0
      vz > 14460.-1865.*(16500.-vx)/(3200.) && vz <= 14460.) return Air;
  
  if (vx > 13300. && vx < 20500.+3050. && //TX54 galerie
      vz > 14460. && vz < 16260. && //estimated wall thickness 1m
      vy > -8680. && vy < -1507.) return Air;
  
  if (vx > 26600-6050. && vx < 20500.+3050. && //TX54 going up in sewrvice cavern
      vz > 14460. && vz < 16260. && //estimated wall thickness 1m
      vy >= -1507. && vy < 1000.) return Air;
  
  //R56, LHC beam tunnel East
  if (vz > -85000. && vz <= -29278. && //UJ57 junction to UXC55 cavern
      sqrt(vy*vy + (vx-350.)*(vx-350.)) < 1900. &&
      vy > -1000.) return Air;
  
  //R54, LHC beam tunnel West
  if (vz >= 29278. && vz < 63000. && //UJ57 junction to UXC55 cavern
      sqrt(vy*vy + (vx-350.)*(vx-350.)) < 1900. &&
      vy > -1000.) return Air;
  
  //UJ56 cavern
  if (vz > -58875. && vz < -33927. &&
      sqrt(vy*vy + (vx-4450.)*(vx-4450.)) < 6750. && vy > -1000. && //and beam shielding
      !(vx > 2250. && vx < 4250. && (vz > -(33927.+18000.) || vz < -(33927.+19500.))) && 
      !(vx >= 4250. && vx < 6650. && vz > -(33927.+18000.) && vz < -(33927.+16000.)))
    return Air;
  
  //connection between PM56 shaft and UJ56 cavern
  if (vx > 9000. && vx < 18220. &&
      sqrt((vy-50.)*(vy-50.)+(vz+24227.-Z_PX56)*(vz+24227.-Z_PX56)) < 3550.
      && vy > -1000.) return Air;
  
  return Unknown;
  
}


inline int inWallAfterAir(double vx, double vy, double vz) {
  // phase II surface building
  if (vy < SurfaceOfEarth && vy >= (SurfaceOfEarth-2250.)) {
    if (std::fabs(vz-Z_PX56) < 30000. && std::fabs(vx) < 10950) return Wall;
    // foundation of crane
    if (std::fabs(vz-Z_PX56) < 9000. && std::fabs(vx) >= 10950 && std::fabs(vx) < 16950) 
      return Wall;
  }
  
  // CMS cavern (UXC 55)
  if (std::fabs(vz) < 29278. && sqrt((vx*1.1576)*(vx*1.1576) + vy*vy) < 16830. &&
      vy > -11762.) return Wall;
  
  // access shaft (PX 56)
  if (vy > 0. && vy < (SurfaceOfEarth-2250.) && //t(shaft wall)=2150.
      sqrt(vx*vx + (vz-Z_PX56)*(vz-Z_PX56)) < 12400.) return Wall;
  
  // Shaft (PM 54)
  if (vy > 3233. && vy < (SurfaceOfEarth-1000.) && //t~=t(PX56)/R(PX56)*R(PM54)
      sqrt((vx-26600.)*(vx-26600.) + (vz-30100.-Z_PX56)*(vz-30100.-Z_PX56)) 
      < 6050.+2150./10250.*6050.) return Wall;
  //sqrt((vx-5000.)*(vx-5000.) + (vz-30100.-Z_PX56)*(vz-30100.-Z_PX56))//@@@@@ 
  //< 10050.+2150./10250.*6050.) return Wall;//@@@@@
  else if (vy >= SurfaceOfEarth-1000. && vy < SurfaceOfEarth && //t~=t(PX56)/R(PX56)*R(PM54)
	   sqrt((vx-26600.)*(vx-26600.) + (vz-30100.-Z_PX56)*(vz-30100.-Z_PX56)) 
	   < 6050.+2150./10250.*6050. +1800.) return Wall;
  //sqrt((vx-5000.)*(vx-5000.) + (vz-30100.-Z_PX56)*(vz-30100.-Z_PX56))//@@@@@ 
  //< 10050.+2150./10250.*10050. +1800.) return Wall; //@@@@@@
  
  // Shaft (PM 56)
  if (vy > -5450. && vy < (SurfaceOfEarth-1000.) && //t~=t(PX56)/R(PX56)*R(PM56)
      sqrt((vx-18220.)*(vx-18220.) + (vz+24227.-Z_PX56)*(vz+24227.-Z_PX56)) 
      < 3550.+2150./10250.*3550.) return Wall;
  else if (vy > SurfaceOfEarth-1000. && vy < SurfaceOfEarth && //t~=t(PX56)/R(PX56)*R(PM56)
	   sqrt((vx-18220.)*(vx-18220.) + (vz+24227.-Z_PX56)*(vz+24227.-Z_PX56)) 
	   < 3550.+2150./10250.*3550. +1800.) return Wall;
  
  //Service cavern USC 55
  if (vz > -(22050.+1150.) && vz < (62050.+1150.) &&
      sqrt((vx-29550.)*(vx-29550.) + (vy-3233.)*(vy-3233.)) < 9050.+950. &&
      vy > -3650.-2000.) return Wall; //8762=estimate, to be checked 
  
  //Pillar between CMS collision cavern and service cavern
  if (vz > -29278.+1000. && vz < 29278.+1000.) {
    if (vy > -17985. && vy < 10410. && vx > 13300. && vx < 20500.) 
      return Wall;
    
    if (vy > 0. && vy < 10410. && vx > 10000. && vx <= 13300.)
      return Wall;
    
    if (vy > -3650.-2000. && vy < -3233. && // bottom edge between pillar and service cavern
	vx > 20000. && vx < 24000.) return Wall;
    if (vy > -11762. && vy < -5000. && // bottom edge between pillar and UXC55 cavern
	vx > 10500. && vx < 14000.) return Wall;
  }
  
  if (vy > -14000. && vy < -1450. && //TX54 galerie surrounding
      vz > 13460. && vz < 17260. && vx >= 20500. && vx < 24550.) 
    return Wall;
  
  //R56, LHC beam tunnel East
  if (vz > -85000. && vz < -28510.) { //UJ57 junction to UXC55 cavern
    if (sqrt(vy*vy + (vx-350.)*(vx-350.)) < 2250.) return Wall;
  }
  
  //R54, LHC beam tunnel West
  if (vz > 26550. && vz < 63000. && //UJ57 junction to UXC55 cavern
      sqrt(vy*vy + (vx-350.)*(vx-350.)) < 2250.) return Wall;
  
  //UJ56 cavern
  if (vz > -(58875.+500.) && vz < -(33927.-500.) &&
      sqrt(vy*vy + (vx-4450.)*(vx-4450.)) < (6750.+500.) && vy > -3650.) 
    return Wall;
  
  //connection between PM56 shaft and UJ56 cavern
  if (vx > 9000. && vx < 18220. &&
      sqrt((vy-50.)*(vy-50.)+(vz+24227.-Z_PX56)*(vz+24227.-Z_PX56)) < 3550.+500. 
      && vy > -3650.) return Wall;
  
  return Unknown;
}



inline int inClayOrRockAfterWall(double vx, double vy, double vz, double ClayWidth) {
  
  //So, it is not plug, air and wall, Check for clay
  if (vy >= SurfaceOfEarth - ClayWidth && vy < SurfaceOfEarth)
    return Clay;
  
  //So, it is not plug, air, wall and clay, Check for rock
  if (vy < SurfaceOfEarth - ClayWidth)
    return Rock;
  
  return Unknown;
  
}



inline int inClayAfterWall(double vx, double vy, double vz, double ClayWidth) {
  
  //So, it is not plug, air and wall, Check for clay
  if (vy >= SurfaceOfEarth - ClayWidth && vy < SurfaceOfEarth)
    return Clay;
  
  return Unknown;
  
}



inline int inRockAfterClay(double vx, double vy, double vz, double ClayWidth) {
  
  //So, it is not plug, air, wall and clay, Check for rock
  if (vy < SurfaceOfEarth - ClayWidth)
    return Rock;
  
  return Unknown;
  
}





inline int inMat(double vx, double vy, double vz,
		 double PlugVx = PlugOnShaftVx, double PlugVz = PlugOnShaftVz,
		 double ClayWidth = DefaultClayWidth) {
  
  //check for Plug
  if (inPlug(vx, vy, vz, PlugVx, PlugVz)) return Plug;
  
  //So, it is not plug, Check for air
  if (inAirAfterPlug(vx, vy, vz)) return Air;
  
  //So, it is not plug and air, Check for wall
  if (inWallAfterAir(vx, vy, vz)) return Wall;
  
  //So, it is not plug, air and wall, Check for clay
  if (vy >= SurfaceOfEarth - ClayWidth && vy < SurfaceOfEarth)
    return Clay;
  
  //So, it is not plug, air, wall and clay, Check for rock
  if (vy < SurfaceOfEarth - ClayWidth)
    return Rock;
  
  
  std::cout << "Point5MaterialMap.h: Warning! No Material recognised for point: vx="
	    << vx << " vy=" << vy << " vz=" << vz << std::endl;
  //Something went wrong
  return Unknown;
  
}




#endif
