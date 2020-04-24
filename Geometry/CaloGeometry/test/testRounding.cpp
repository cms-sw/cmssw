#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <cstdio>
#include <iostream>
#include <cmath>

struct HFCellParameters
{
  HFCellParameters( int f_eta, int f_depth, int f_phiFirst, int f_phiStep, int f_dPhi,
		    float f_zMin, float f_zMax, float f_rMin, float f_rMax )
    : eta( f_eta ),
      depth( f_depth ),
      phiFirst( f_phiFirst ),
      phiStep( f_phiStep ),
      dphi( f_dPhi ),
      zMin( f_zMin ),
      zMax( f_zMax ),
      rMin( f_rMin ),
      rMax( f_rMax )
    {}
 
  int eta;
  int depth;
  int phiFirst;
  int phiStep;
  int dphi;
  float zMin;
  float zMax;
  float rMin;
  float rMax;
};

int main()
{
  const float HFZMIN1 = 1115.;
  const float HFZMIN2 = 1137.;
  const float HFZMAX = 1280.1;
  const int   MAX_HCAL_PHI = 72;
  const float DEGREE2RAD = M_PI / 180.;
    
  HFCellParameters cells [] = {
    // eta, depth, firstPhi, stepPhi, deltaPhi, zMin, zMax, rMin, rMax
    HFCellParameters (29, 1, 1, 2, 10, HFZMIN1, HFZMAX,116.2,130.0),
    HFCellParameters (29, 2, 1, 2, 10, HFZMIN2, HFZMAX,116.2,130.0),
    HFCellParameters (30, 1, 1, 2, 10, HFZMIN1, HFZMAX, 97.5,116.2),
    HFCellParameters (30, 2, 1, 2, 10, HFZMIN2, HFZMAX, 97.5,116.2),
    HFCellParameters (31, 1, 1, 2, 10, HFZMIN1, HFZMAX, 81.8, 97.5),
    HFCellParameters (31, 2, 1, 2, 10, HFZMIN2, HFZMAX, 81.8, 97.5),
    HFCellParameters (32, 1, 1, 2, 10, HFZMIN1, HFZMAX, 68.6, 81.8),
    HFCellParameters (32, 2, 1, 2, 10, HFZMIN2, HFZMAX, 68.6, 81.8),
    HFCellParameters (33, 1, 1, 2, 10, HFZMIN1, HFZMAX, 57.6, 68.6),
    HFCellParameters (33, 2, 1, 2, 10, HFZMIN2, HFZMAX, 57.6, 68.6),
    HFCellParameters (34, 1, 1, 2, 10, HFZMIN1, HFZMAX, 48.3, 57.6),
    HFCellParameters (34, 2, 1, 2, 10, HFZMIN2, HFZMAX, 48.3, 57.6),
    HFCellParameters (35, 1, 1, 2, 10, HFZMIN1, HFZMAX, 40.6, 48.3),
    HFCellParameters (35, 2, 1, 2, 10, HFZMIN2, HFZMAX, 40.6, 48.3),
    HFCellParameters (36, 1, 1, 2, 10, HFZMIN1, HFZMAX, 34.0, 40.6),
    HFCellParameters (36, 2, 1, 2, 10, HFZMIN2, HFZMAX, 34.0, 40.6),
    HFCellParameters (37, 1, 1, 2, 10, HFZMIN1, HFZMAX, 28.6, 34.0),
    HFCellParameters (37, 2, 1, 2, 10, HFZMIN2, HFZMAX, 28.6, 34.0),
    HFCellParameters (38, 1, 1, 2, 10, HFZMIN1, HFZMAX, 24.0, 28.6),
    HFCellParameters (38, 2, 1, 2, 10, HFZMIN2, HFZMAX, 24.0, 28.6),
    HFCellParameters (39, 1, 1, 2, 10, HFZMIN1, HFZMAX, 20.1, 24.0),
    HFCellParameters (39, 2, 1, 2, 10, HFZMIN2, HFZMAX, 20.1, 24.0),
    HFCellParameters (40, 1, 3, 4, 20, HFZMIN1, HFZMAX, 16.9, 20.1),
    HFCellParameters (40, 2, 3, 4, 20, HFZMIN2, HFZMAX, 16.9, 20.1),
    HFCellParameters (41, 1, 3, 4, 20, HFZMIN1, HFZMAX, 12.5, 16.9),
    HFCellParameters (41, 2, 3, 4, 20, HFZMIN2, HFZMAX, 12.5, 16.9)
  };

  for(const auto & param : cells)
  {
    for( int iPhi = param.phiFirst; iPhi <= MAX_HCAL_PHI; iPhi += param.phiStep )
    {
      for( int iside = -1; iside <= 1; iside += 2 ) // both detector sides are identical
      {
	float phiCenter = (( iPhi - 1 ) * 360. / MAX_HCAL_PHI + 0.5 * param.dphi ) * DEGREE2RAD; // middle of the cell
	GlobalPoint inner( param.rMin, 0., param.zMin );
	GlobalPoint outer( param.rMax, 0., param.zMin );
	float etaCenter = 0.5 * ( inner.eta() + outer.eta());
	float perp = param.zMin / sinh ( etaCenter );
	float x = perp * cos( phiCenter );
	float y = perp * sin( phiCenter );
	float z = iside * param.zMin;
	
	// make cell geometry
	GlobalPoint refPoint( x, y, z ); // center of the cell's face
	
	double refEta = refPoint.eta();
	double calcEta = -log( tan( refPoint.theta()/2.));
	double eps = 1.e-5;

	bool same( true );
	same = same && ( fabs( refEta - calcEta ) < eps );
	
	std::cout << "(x, y, z) = (" << x << ", " << y << ", " << z
		  << "), theta = " << refPoint.theta() << ", eta = " << refEta
		  << " vs re-calc eta = " << calcEta;
	same ? ( std::cout << std::endl ) : ( std::cout << " DIFFER " << std::endl );
      }
    }
  }
  
  return 0;
}
