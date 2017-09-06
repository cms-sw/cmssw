////////////////////////////////////////////////////////////////////////////////
// File: EcalPreshowerNumberingScheme.cc
// Description: Numbering scheme for preshower detector
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include <iostream>
using namespace std;

EcalPreshowerNumberingScheme::EcalPreshowerNumberingScheme() : 
  EcalNumberingScheme() {

  // For SFLX2a, we use copy# 1-3 
  int vL2ax[3] = { 4,  6, 10};
  int vL2ay[3] = {29, 31, 35};
  // For SFLX2b, we use copy# 1
  int vL2bx[1] = {2};
  int vL2by[1] = {25};
  // For SFLX3a, we use copy# 4-6
  int vL3ax[3] = {30, 34, 36};
  int vL3ay[3] = {35, 31, 29};
  // For SFLX3b, we use copy# 2
  int vL3bx[1] = {38};
  int vL3by[1] = {25};
  // For SFLX1a, we use odd number in copy# 1-52
  int vL1ax[26] = { 2,  4,  4,  8,  8,  8,  8, 10, 12, 12, 14, 14, 20, 20, 26, 26, 28, 28, 30, 32, 32, 32, 32, 36, 36, 38};
  int vL1ay[26] = {21, 25, 21, 33, 29, 25, 21, 21, 25, 21, 31, 27, 32, 28, 31, 27, 25, 21, 21, 33, 29, 25, 21, 25, 21, 21};
  // For SFLX1b, we use copy# 2
  int vL1bx[1] = {22};
  int vL1by[1] = {27};
  // For SFLX1c, we use copy# 1
  int vL1cx[1] = {18};
  int vL1cy[1] = {27};
  // For SFLX1d, we use copy# 2
  int vL1dx[1] = {26};
  int vL1dy[1] = {23};
  // For SFLX1e, we use copy# 1
  int vL1ex[1] = {14};
  int vL1ey[1] = {23};
  // For SFLX0a, we use odd number if copy# 1-46
  int vL0ax[23] = { 6,  6, 10, 10, 12, 12, 14, 16, 16, 18, 18, 20, 22, 22, 24, 24, 26, 28, 28, 30, 30, 34, 34};
  int vL0ay[23] = {26, 21, 30, 25, 34, 29, 35, 36, 31, 36, 31, 36, 36, 31, 36, 31, 35, 34, 29, 30, 25, 26, 21};
  // For SFL0b, we use copy# 2
  int vL0bx[1] = {24};
  int vL0by[1] = {26};
  // For SFL0c, we use copy# 1
  int vL0cx[1] = {16};
  int vL0cy[1] = {26};

  for (int i=0; i<1; ++i) {
    L1bx[i] = vL1bx[i];
    L1by[i] = vL1by[i];
    L1cx[i] = vL1cx[i];
    L1cy[i] = vL1cy[i];
    L1dx[i] = vL1dx[i];
    L1dy[i] = vL1dy[i];
    L1ex[i] = vL1ex[i];
    L1ey[i] = vL1ey[i];
    L0bx[i] = vL0bx[i];
    L0by[i] = vL0by[i];
    L0cx[i] = vL0cx[i];
    L0cy[i] = vL0cy[i];
    L3bx[i] = vL3bx[i];
    L3by[i] = vL3by[i];
    L2bx[i] = vL2bx[i];
    L2by[i] = vL2by[i];
  }

  for (int i=0; i<3; ++i) {
    L3ax[i] = vL3ax[i];
    L3ay[i] = vL3ay[i];
    L2ax[i] = vL2ax[i];
    L2ay[i] = vL2ay[i];
  }

  for (int i=0; i<23; ++i) {
    L0ax[i] = vL0ax[i];
    L0ay[i] = vL0ay[i];
  }

  for (int i=0; i<26; ++i) {
    L1ax[i] = vL1ax[i];
    L1ay[i] = vL1ay[i];
  }

  edm::LogInfo("EcalGeom") << "Creating EcalPreshowerNumberingScheme";
}

EcalPreshowerNumberingScheme::~EcalPreshowerNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Deleting EcalPreshowerNumberingScheme";
}

uint32_t EcalPreshowerNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const  {

  int level = baseNumber.getLevels();
  uint32_t intIndex = 0; 
  if (level > 0) {
    
    // depth index - silicon layer 1-st or 2-nd
    int layer = 0;
    if(baseNumber.getLevelName(0) == "SFSX") {
      layer = 1;
    } else if (baseNumber.getLevelName(0) == "SFSY") {
      layer = 2;
    } else {
      edm::LogWarning("EcalGeom") << "EcalPreshowerNumberingScheme: Wrong name"
				  << " of Presh. Si. Strip : " 
				  << baseNumber.getLevelName(0);
    }
    
    // Z index +Z = 1 ; -Z = 2
    int zside   = baseNumber.getCopyNumber("EREG");
    zside=2*(1-zside)+1;

    // box number
    int box = baseNumber.getCopyNumber(2);

    int x=0,y=0,ix,iy,id;
    int mapX[10] ={0,0,0,0,0,0,0,0,0,0};  int mapY[10] ={0,0,0,0,0,0,0,0,0,0};
    const std::string& ladd = baseNumber.getLevelName(3);
    int ladd_copy = baseNumber.getCopyNumber(3);
    
    if(ladd=="SFLX0a" || ladd=="SFLY0a" ) { 
      mapX[5] = mapX[6] = mapX[7] = mapX[8] = mapX[9] = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 4; 
      mapY[5] = 0; mapY[6] = 1; mapY[7] = 2;  mapY[8] = 3;  mapY[9] = 4; 

      id = (int) ((float)ladd_copy/2+0.5);

      x = L0ax[id-1] + mapX[box-1000-1];
      y = L0ay[id-1] + mapY[box-1000-1];    

      if ((ladd_copy%2) == 0) {
	if (mapX[box-1000-1]==0) x += 1;
	else if (mapX[box-1000-1]==1) x -= 1; 
	y = 41 - y;
      }
    }
    if(ladd=="SFLX0b" || ladd=="SFLY0b") { 
      mapX[4] = mapX[5] = mapX[6] = mapX[7] = mapX[8] = 1; 
      mapY[0] = 1; mapY[1] = 2; mapY[2] = 3; mapY[3] = 4; mapY[4] = 0; 
      mapY[5] = 1; mapY[6] = 2; mapY[7] = 3;  mapY[8] = 4;  mapY[9] = 0; 

      x = L0bx[0] + mapX[box-2000-1];
      y = L0by[0] + mapY[box-2000-1];    

      if (ladd_copy == 1) {
	x = 41 - x;
	y = 41 - y;
      }
    }
    if(ladd=="SFLX0c" || ladd=="SFLY0c") {
      mapX[5] = mapX[6] = mapX[7] = mapX[8] = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 4; 
      mapY[5] = 1; mapY[6] = 2; mapY[7] = 3;  mapY[8] = 4;  mapY[9] = 0; 

      x = L0cx[0] + mapX[box-3000-1];
      y = L0cy[0] + mapY[box-3000-1]; 

      if (ladd_copy == 2) {
	x = 41 - x;
	y = 41 - y;
      }
    }
    if(ladd=="SFLX1a" || ladd=="SFLY1a" ) {
      mapX[4] = mapX[5] = mapX[6] = mapX[7]  = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 0; 
      mapY[5] = 1; mapY[6] = 2; mapY[7] = 3;  mapY[8] = 0;  mapY[9] = 0; 

      id = (int) ((float)ladd_copy/2+0.5);

      x = L1ax[id-1] + mapX[box-4000-1];
      y = L1ay[id-1] + mapY[box-4000-1];    

      if ((ladd_copy%2) == 0) {
	if (mapX[box-4000-1]==0) x += 1;
	else if (mapX[box-4000-1]==1) x -= 1; 
	y = 41 - y;
      }
    }
    if(ladd=="SFLX1b" || ladd=="SFLY1b" ) {
      mapX[3] = mapX[4] = mapX[5] = mapX[6]  = 1; 
      mapY[0] = 1; mapY[1] = 2; mapY[2] = 3; mapY[3] = 0; mapY[4] = 1; 
      mapY[5] = 2; mapY[6] = 3; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 

      x = L1bx[0] + mapX[box-5000-1];
      y = L1by[0] + mapY[box-5000-1];    

      if (ladd_copy == 1) {
	x = 41 - x;
	y = 41 - y;
      }
    }
    if(ladd=="SFLX1c" || ladd=="SFLY1c" ) {
      mapX[4] = mapX[5] = mapX[6]  = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 1; 
      mapY[5] = 2; mapY[6] = 3; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 

      x = L1cx[0] + mapX[box-6000-1];
      y = L1cy[0] + mapY[box-6000-1];    

      if (ladd_copy == 2) {
	x = 41 - x;
	y = 41 - y;
      }
    }
    if(ladd=="SFLX1d" || ladd=="SFLY1d" ) {
      mapX[2] = mapX[3] = mapX[4] = mapX[5]  = 1; 
      mapY[0] = 2; mapY[1] = 3; mapY[2] = 0; mapY[3] = 1; mapY[4] = 2; 
      mapY[5] = 3; mapY[6] = 0; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 

      x = L1dx[0] + mapX[box-7000-1];
      y = L1dy[0] + mapY[box-7000-1];    

      if (ladd_copy == 1) {
	x = 41 - x;
	y = 41 - y;
      }
    }
    if(ladd=="SFLX1e" || ladd=="SFLY1e") {
      mapX[4] = mapX[5] = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 2; 
      mapY[5] = 3; mapY[6] = 0; mapY[7] = 0; mapY[8] = 0; mapY[9] = 0; 

      x = L1ex[0] + mapX[box-8000-1];
      y = L1ey[0] + mapY[box-8000-1];    

      if (ladd_copy == 2) {
	x = 41 - x;
	y = 41 - y;
      }
    }
    if(ladd=="SFLX3a" || ladd=="SFLY3a" ) {
      mapX[4] = mapX[5] = mapX[6] = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 0; 
      mapY[5] = 1; mapY[6] = 2; mapY[7] = 0; mapY[8] = 0; mapY[9] = 0; 

      id = (ladd_copy>3)? ladd_copy-3 : 4-ladd_copy;     

      x = L3ax[id-1] + mapX[box-9000-1];
      y = L3ay[id-1] + mapY[box-9000-1];    

      if (ladd_copy<4) {
	x = 41 - x;
	y = 41 - y;
      }
    }
    if(ladd=="SFLX3b" || ladd=="SFLY3b") {
      mapX[4] = mapX[5] = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 0; 
      mapY[5] = 1; mapY[6] = 0; mapY[7] = 0; mapY[8] = 0; mapY[9] = 0; 

      x = L3bx[0] + mapX[box-11000-1];
      y = L3by[0] + mapY[box-11000-1];

      if (ladd_copy == 1) {
	x = 41 - x;
	y = 41 - y;
      }  
    }
    if(ladd=="SFLX2a" || ladd=="SFLY2a") {
      mapX[3] = mapX[4] = mapX[5] = mapX[6]  = 1; 
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 0; mapY[4] = 1; 
      mapY[5] = 2; mapY[6] = 3; mapY[7] = 0; mapY[8] = 0; mapY[9] = 0; 

      id = (ladd_copy>3)? 7-ladd_copy : ladd_copy;     

      x = L2ax[id-1] + mapX[box-10000-1];
      y = L2ay[id-1] + mapY[box-10000-1];

      if (ladd_copy>3) {
	x = 41 - x;
	y = 41 - y;
      }  
    }
    if(ladd=="SFLX2b" || ladd=="SFLY2b") {
      mapX[2] = mapX[3] = mapX[4] = mapX[5]  = 1;             
      mapY[0] = 0; mapY[1] = 1; mapY[2] = 0; mapY[3] = 1; mapY[4] = 2; 
      mapY[5] = 3; mapY[6] = 0; mapY[7] = 0; mapY[8] = 0; mapY[9] = 0; 

      x = L2bx[0] + mapX[box-12000-1];
      y = L2by[0] + mapY[box-12000-1];

      if (ladd_copy == 2) {
	x = 41 - x;
	y = 41 - y;
      }  
    }
       
    if (zside<0 && layer == 1) x = 41 - x;

    ix = x;
    iy = y;

    if (layer == 2) {
      x = (zside>0) ? iy : 41 - iy; 
      y = 41 - ix;
    }

    // strip number inside wafer
    int strip = baseNumber.getCopyNumber(0);
    
    if (layer == 1) {
      if (zside>0 && y<=20) 
	strip = 33 - strip;
      else if (zside<0 && y>20)
	strip = 33 - strip;
    } else if (layer == 2) {
      if (zside>0 && x<=20) 
	strip = 33 - strip;
      else if (zside<0 && x>20)
	strip = 33 -strip;
    }
    
    intIndex =  ESDetId(strip, x, y, layer, zside).rawId(); 
    
    LogDebug("EcalGeom") << "EcalPreshowerNumberingScheme : zside "<<zside<<" Ladd "<< ladd << " ladd_copy: "<<ladd_copy<<" box "<<box<<" x "<<x<<" y "<<y<<" layer "<<layer<<" strip " << strip<<" UnitID 0x" << std::hex << intIndex << std::dec;
    
    for (int ich = 0; ich < level; ich++) {
      LogDebug("EcalGeom") << "Name = " << baseNumber.getLevelName(ich) 
			   << " copy = " << baseNumber.getCopyNumber(ich);
    }
  }
  
  return intIndex;
}

