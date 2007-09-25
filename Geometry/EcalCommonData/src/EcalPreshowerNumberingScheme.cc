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

   int iquad_M[40] = {5,7,10,11,13,13,14,15,16,17,17,17,18,
		     19,19,19,19,19,19,19,19,19,19,19,19,19,19,
		     18,17,17,17,16,15,
		     14,13,13,11,10,7,5};
  int iquad_m[40] = {1,1,1,1,1,1,1,1,1,1,1,1,1,
		     4,4,6,6,8,8,8,8,8,8,6,6,4,4,
		     1,1,1,1,1,1,1,1,1,1,1,1,1};

  int i;
  for(i=0; i<40; i++) {
    iquad_max[i] = iquad_M[i];
    iquad_min[i] = iquad_m[i];
  }
  
   Ncols[0] = 2*(iquad_max[0]-iquad_min[0]+1);
  Ntot[0] = Ncols[0];
  for (i=1; i<40; i++) {
    Ncols[i] = iquad_max[i]-iquad_min[i]+1;
    Ntot[i] = Ntot[i-1]+2*Ncols[i];
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
    // wafer number
    int wafer = baseNumber.getCopyNumber(2);
    int x,y,ix,iy,start=-1;
    int mapX[10] ={0,0,0,0,0,0,0,0,0,0};  int mapY[10] ={0,0,0,0,0,0,0,0,0,0};
    std::string ladd = baseNumber.getLevelName(3);
    int ladd_copy = baseNumber.getCopyNumber(3);

int LXV[154] = {4,14,4,19,10,8,8,15,10,6,10,11,12,5,14,4,14,9,16,4,16,9,18,4, 
                 20,4,20,9,22,4,22,9,24,5,26,6,26,11,28,10,28,15,32,14,32,19,
                 0,19,2,15,2,19,6,7,6,11,6,15,6,19,8,19,10,15,10,19,12,9,12,13,19,8,19,12,
                 24,9,24,13,26,15,26,19,28,19,30,7,30,11,30,15,30,19,34,15,34,19,26,19,
                 22,14,15,25,
                 14,14,23,25,
                 20,13,17,25,
                 16,13,21,25,
                 24,17,13,25,
                 12,17,25,25,
                 1,24,3,28,5,30,9,24,28,5,32,9,34,11,36,15,
                 0,15,2,11,4,9,8,5,29,34,33,20,35,28,37,24};

if(ladd=="SFLX0a" || ladd=="SFLY0a" ) { start = 0;  
mapX[5] = mapX[6] = mapX[7] = mapX[8] = mapX[9] = 1; 
mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 4; 
mapY[5] = 0; mapY[6] = 1; mapY[7] = 2;  mapY[8] = 3;  mapY[9] = 4; 
		}
if(ladd=="SFLX0b" || ladd=="SFLY0b") { start = 98;  
mapX[4] = mapX[5] = mapX[6] = mapX[7] = mapX[8] = 1; 
mapY[0] = 1; mapY[1] = 2; mapY[2] = 3; mapY[3] = 4; mapY[4] = 0; 
mapY[5] = 1; mapY[6] = 2; mapY[7] = 3;  mapY[8] = 4;  mapY[9] = 0; 
	}
if(ladd=="SFLX0c" || ladd=="SFLY0c") {start = 102;  
mapX[5] = mapX[6] = mapX[7] = mapX[8] = 1; 
mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 4; 
mapY[5] = 1; mapY[6] = 2; mapY[7] = 3;  mapY[8] = 4;  mapY[9] = 0; 
  }
if(ladd=="SFLX1a" || ladd=="SFLY1a" ) {start = 46;  
 mapX[4] = mapX[5] = mapX[6] = mapX[7]  = 1; 
 mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 0; 
mapY[5] = 1; mapY[6] = 2; mapY[7] = 3;  mapY[8] = 0;  mapY[9] = 0; 
	}
if(ladd=="SFLX1b" || ladd=="SFLY1b" ) {start = 106;   
 mapX[3] = mapX[4] = mapX[5] = mapX[6]  = 1; 
 mapY[0] = 1; mapY[1] = 2; mapY[2] = 3; mapY[3] = 0; mapY[4] = 1; 
mapY[5] = 2; mapY[6] = 3; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 
	}
if(ladd=="SFLX1c" || ladd=="SFLY1c" ) {start = 116; 
 mapX[4] = mapX[5] = mapX[6]  = 1; 
 mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 1; 
mapY[5] = 2; mapY[6] = 3; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 
	}
if(ladd=="SFLX1d" || ladd=="SFLY1d" ) {start = 114;
 mapX[2] = mapX[3] = mapX[4] = mapX[5]  = 1; 
 mapY[0] = 2; mapY[1] = 3; mapY[2] = 0; mapY[3] = 1; mapY[4] = 2; 
mapY[5] = 3; mapY[6] = 0; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 
	}
if(ladd=="SFLX1e" || ladd=="SFLY1e") {start = 118; 
 mapX[4] = mapX[5] = 1; 
 mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 2; 
mapY[5] = 3; mapY[6] = 0; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 
	}
if(ladd=="SFLX2" || ladd=="SFLY2" ) {start = 122; 
 mapX[4] = mapX[5] = mapX[6] = 1; 
 mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 3; mapY[4] = 1; 
mapY[5] = 2; mapY[6] = 3; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 
	}
if(ladd=="SFLX3" || ladd=="SFLY3") {start = 138; 
 mapX[3] = mapX[4] = mapX[5] = mapX[6]  = 1; 
 mapY[0] = 0; mapY[1] = 1; mapY[2] = 2; mapY[3] = 0; mapY[4] = 1; 
mapY[5] = 2; mapY[6] = 3; mapY[7] = 0;  mapY[8] = 0;  mapY[9] = 0; 
	}

x=LXV[ladd_copy-1+start]+mapX[wafer-1];
y=LXV[ladd_copy-1+start+1]-mapY[wafer-1];

if (ladd_copy%2==0) x=LXV[ladd_copy-2+start]+1-mapX[wafer-1];
if (ladd_copy%2==0) y=39 - LXV[ladd_copy-2+start+1]+mapY[wafer-1];

if(start>97) x=LXV[2*ladd_copy-2+start]+mapX[wafer-1];
if(start>97) y=LXV[2*ladd_copy-2+start+1]-mapY[wafer-1];

 if (layer==2) {
    ix = 39 - x;
    iy = 39 - y;
    x = iy;
    y = ix;
  }


x++;
y=( 39 - y ) +1; //To be compliant with CMSSW numbering scheme 

 //   findXY(layer, wafer, x, y);
    // strip number inside wafer
    int strip = baseNumber.getCopyNumber(0);

  if (y>20) {
      strip = 33 - strip;
    }

    if ( zside < 0 ) {
      x=41-x;
      if (layer == 1)
	strip = 33 - strip;
    }


    // std::cout << "End: Ladd: "<< ladd << " ladd_copy: "<<ladd_copy<<" wafer: "<<wafer<<" start: "<<start<<" x: "<<x<<" y: "<<y<<" layer: "<<layer<<" strip " << strip<<std::endl;
    
    intIndex =  ESDetId(strip, x, y, layer, zside).rawId(); 
    
    LogDebug("EcalGeom")  << "EcalPreshowerNumberingScheme : zside " << zside 
			 << " layer " << layer << " wafer " << wafer << " X " 
			 << x << " Y "<< y << " strip " << strip 
			 << " UnitID 0x" << std::hex << intIndex << std::dec;
    for (int ich = 0; ich < level; ich++) {
     LogDebug("EcalGeom") << "Name = " << baseNumber.getLevelName(ich) 
			   << " copy = " << baseNumber.getCopyNumber(ich);
    }
  }

  return intIndex;
}

void EcalPreshowerNumberingScheme::findXY(const int& layer, const int& waf, int& x, int& y)  const {

  int i,ix,iy;

  y = 0;
  for (i=0; i<40; i++) {
    if (waf > Ntot[i]) y = i + 1;
  }

  x = 19 - iquad_max[y];
  x = (y==0) ? (x + waf) : (x + waf - Ntot[y-1]);

  int iq = iquad_min[y];
  if (iq != 1 && x > (Ncols[y]+19-iquad_max[y])) {
    x = x + (iq-1)*2;
  }
  
  if (layer==2) {
    ix = 39 - x;
    iy = 39 - y;
    x = iy;
    y = ix;
  }
  
  x++;
  y=( 39 - y ) +1; //To be compliant with CMSSW numbering scheme 
}
