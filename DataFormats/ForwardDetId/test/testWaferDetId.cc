#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

int nCorner(double r, double R, double rin, double rout, double xpos, 
	    double ypos) {
  int corner(0);
  double xc[6], yc[6];
  xc[0] = xpos;    yc[0] = ypos+R;
  xc[1] = xpos-r;  yc[1] = ypos+0.5*R;
  xc[2] = xpos-r;  yc[2] = ypos-0.5*R;
  xc[3] = xpos;    yc[3] = ypos-R;
  xc[4] = xpos+r;  yc[4] = ypos-0.5*R;
  xc[5] = xpos+r;  yc[5] = ypos+0.5*R;
  for (int k=0; k<6; ++k) {
    double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
    if (rpos >= rin && rpos <= rout) ++corner;
  }
  return corner;
}

void testWaferDetId(WaferDetId const& id) {
  std::cout << "                 WaferDetId::EE:HE= " << id.isEE() << ":"
	    << id.isHE() << " type= " << id.type()  << " z= " << id.zside() 
	    << " layer= " << id.layer() 
	    << " wafer(u,v:x,y)= (" << id.waferU() << "," << id.waferV() << ":"
	    << id.waferX() << "," << id.waferY() << ")"
	    << " cell(u,v:x,y)= (" << id.cellU() << "," << id.cellV() << ":"
	    << id.cellX() << "," << id.cellY() << ")" << std::endl;
}

void testHGCalWafer(int layer, double rin, double rout) {
  const double waferSize(167.4408);
  const double rMaxFine(750.0), rMaxMiddle(1200.0);
  const int    zside(1), cellu(0), cellv(0);
  const std::string waferType[2] = {"Virtual", "Real   "};

  std::cout << "\n\nHGCalEE::Layer " << layer << " R-range " << rin << ":"
	    << rout << std::endl;
  double r     = 0.5*waferSize;
  double R     = 2.0*r/std::sqrt(3.0);
  double dy    = 0.75*R;
  int    N     = (int)(0.5*rout/r) + 2;
  int    nreal(0), nvirtual(0);
  int    ntype[3] = {0,0,0};
  for (int v = -N; v <= N; ++v) {
    for (int u = -N; u <= N; ++u) {
      int nr = 2*v;
      int nc =-2*u+v;
      double xpos = nc*r;
      double ypos = nr*dy;
      int corner = nCorner(r, R, rin, rout, xpos, ypos);
      if (corner > 0) {
	double rr = std::sqrt(xpos*xpos+ypos*ypos);
	int type  = (rr < rMaxFine) ? 0 : ((rr < rMaxMiddle) ? 1 : 2);
	HGCSiliconDetId id(DetId::HGCalEE,zside,type,layer,u,v,cellu,cellv);
	int cornerAll = (corner == 6) ? 1 : 0;
	std::cout << waferType[cornerAll] << " Wafer " << id << std::endl;
	testWaferDetId(id);
	if (corner  == 6) {
	  ++nreal;
	  ++ntype[type];
	} else {
	  ++nvirtual;
	}
      }
    }
  }
  std::cout << nreal << " full wafers of type 0:" << ntype[0] << " 1:"
	    << ntype[1] << " 2:" << ntype[2] << " and " << nvirtual
	    << " partial wafers for r-range " << rin << ":" << rout 
	    << std::endl;
}

void testHFNoseWafer(int layer, double zpos) {
  const double waferSize(167.4408);
  const int    zside(1), cellu(0), cellv(0), type(0);
  const std::string waferType[2] = {"Virtual", "Real   "};

  double rin  = zpos*tan(1.64*3.1415926/180.0);
  double rout = zpos*tan(5.70*3.1415926/180.0);
  std::cout << "\n\nHFNose::Layer " << layer << " Z " << zpos << " R-range "
	    << rin << ":" << rout << std::endl;
  double r     = 0.5*waferSize;
  double R     = 2.0*r/std::sqrt(3.0);
  double dy    = 0.75*R;
  int    N     = (int)(0.5*rout/r) + 2;
  int    nreal(0), nvirtual(0);
  for (int v = -N; v <= N; ++v) {
    for (int u = -N; u <= N; ++u) {
      int nr = 2*v;
      int nc =-2*u+v;
      double xpos = nc*r;
      double ypos = nr*dy;
      int corner = nCorner(r, R, rin, rout, xpos, ypos);
      if (corner > 0) {
	HFNoseDetId id(zside,type,layer,u,v,cellu,cellv);
	int cornerAll = (corner == 6) ? 1 : 0;
	std::cout << waferType[cornerAll] << " Wafer " << id << std::endl;
	testWaferDetId(id);
	if (corner  == 6) {
	  ++nreal;
	} else {
	  ++nvirtual;
	}
      }
    }
  }
  std::cout << nreal << " full, " << nvirtual << " virtual wafers and "
	    << nreal+nvirtual << " of type 0 for r-range " << rin << ":" 
	    << rout << std::endl;
}

int main() {

  testHGCalWafer(1,  319.80, 1544.30);
  testHGCalWafer(28, 352.46, 1658.68);
  testHFNoseWafer(1, 10952.0);
  testHFNoseWafer(8, 11155.0);

  return 0;
}
