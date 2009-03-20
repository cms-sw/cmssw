#include "Geometry/EcalMapping/interface/ESElectronicsMapper.h"

ESElectronicsMapper::ESElectronicsMapper(const edm::ParameterSet& ps) {

  lookup_ = ps.getUntrackedParameter<edm::FileInPath>("LookupTable");

  for (int i=0; i<2; ++i) 
    for (int j=0; j<2; ++j) 
      for (int k=0; k<40; ++k) 
	for (int m=0; m<40; ++m) 
	  fed_[i][j][k][m] = -1; 
 
  // read in look-up table
  int nLines, z, iz, ip, ix, iy, fed, kchip, pace, bundle, fiber, optorx;
  std::ifstream file;
  file.open(lookup_.fullPath().c_str());
  if( file.is_open() ) {

    file >> nLines;

    for (int i=0; i<nLines; ++i) {
      file>> iz >> ip >> ix >> iy >> fed >> kchip >> pace >> bundle >> fiber >> optorx;
      
      if (iz==-1) z = 2;
      else z = iz;
      
      fed_[z-1][ip-1][ix-1][iy-1] = fed + 519;
    }

  } else {
    std::cout<<"ESElectronicsMapper::ESElectronicsMapper : Look up table file can not be found in "<<lookup_.fullPath().c_str()<<std::endl;
  }

}

int ESElectronicsMapper::getFED(const ESDetId& id) { 

  int zside;
  if (id.zside()<0) zside = 2;
  else zside = id.zside();

  return fed_[zside-1][id.plane()-1][id.six()-1][id.siy()-1]; 
} 

int ESElectronicsMapper::getFED(int zside, int plane, int x, int y) { 

  return fed_[zside-1][plane-1][x-1][y-1]; 
} 

std::vector<int> ESElectronicsMapper::GetListofFEDs(const EcalEtaPhiRegion region) const {
  std::vector<int> FEDs;
  GetListofFEDs(region, FEDs);
  return FEDs;
}

void ESElectronicsMapper::GetListofFEDs(const EcalEtaPhiRegion region, std::vector<int> & FEDs) const {

  double etahigh = region.etaHigh();
  double phihigh = region.phiHigh();  
  double etalow = region.etaLow();
  double philow = region.phiLow();

  int zside = (etahigh > 0) ? 1 : 2;

  int x[4][2], y[4][2], fed[4][2];

  // for plane 1
  findXY(1, etahigh, phihigh, x[0][1], y[0][1]);
  findXY(1, etahigh, philow,  x[1][1], y[1][1]);
  findXY(1, etalow,  phihigh, x[2][1], y[2][1]);
  findXY(1, etalow,  philow,  x[3][1], y[3][1]);

  // for plane 2
  findXY(2, etahigh, phihigh, x[0][2], y[0][2]);
  findXY(2, etahigh, philow,  x[1][2], y[1][2]);
  findXY(2, etalow,  phihigh, x[2][2], y[2][2]);
  findXY(2, etalow,  philow,  x[3][2], y[3][2]);
  
  for (int i=0; i<4; ++i) 
    for (int j=0; j<2; ++j) {
      if (x[i][j]<0 || y[i][j]<0) continue;
      fed[i][j] = fed_[zside-1][j][x[i][j]][y[i][j]];
      FEDs.push_back(fed[i][j]);
    }

  sort(FEDs.begin(), FEDs.end());
  unique(FEDs.begin(), FEDs.end());

}

void ESElectronicsMapper::findXY(const int plane, const double eta, const double phi, int &row, int &col) const {
  
  float zplane_[2]= { 303.353, 307.838 };
  
  float waf_w_ = 6.3; 
  float act_w_ = 6.1; 
  float intra_lad_gap_ = 0.04; 
  float inter_lad_gap_ = 0.05;
  float centre_gap_ = 0.05; 

  float theta = 2.*atan(exp(-1.*eta));
  float r = zplane_[plane-1]/cos(theta);
  float x = r * sin(theta) * cos(phi);
  float y = r * sin(theta) * sin(phi);
  
  float x0,y0;

  if (plane == 1) {
    x0 = x;
    y0 = y;
  } else {
    y0 = x;
    x0 = y;
  }

  //find row
  int imul = (y0 < 0.) ? +1 : -1 ; 
  float yr = -(y0 + imul*centre_gap_ )/act_w_;
  row = (yr < 0.) ? (19 + int(yr) ) : (20 + int(yr));
  row= 40 - row;

  if (row < 1 || row > 40 ) row = -1;
  
  //find col
  col = 40 ;
  int nlad = (col < 20 ) ? (20-col)/2 :(19-col)/2 ;
  float edge =  (20-col) * (waf_w_ + intra_lad_gap_)+ nlad * inter_lad_gap_;
  edge = -edge;
  while (x0 < edge && col > 0){
    col--;
    nlad = (col < 20 ) ? (20-col)/2 :(19-col)/2 ;
    edge = (20-col) * (waf_w_ + intra_lad_gap_) + nlad * inter_lad_gap_;   
    edge = -edge;
  }
  
  col++;

  if ( col < 1 || col > 40 || x0 < edge) col = -1;

}
