#ifndef HGCalCommonData_HGCalGeomParameters_h
#define HGCalCommonData_HGCalGeomParameters_h

/** \class HGCalGeomParameters
 *
 * this class extracts some geometry constants from CompactView
 * to be used by Reco Geometry/Topology
 *  
 *  $Date: 2015/06/25 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
 *
 */

#include <string>
#include <vector>
#include <iostream>

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class DDCompactView;    
class DDFilteredView;
class HGCalParameters;

class HGCalGeomParameters {

public:

  HGCalGeomParameters();
  ~HGCalGeomParameters();
  void loadGeometrySquare(const DDFilteredView&, HGCalParameters&,
			  const std::string&);
  void loadGeometryHexagon(const DDFilteredView&, HGCalParameters&,
			   const std::string&, const DDCompactView*,
			   const std::string&, const std::string&);
  void loadSpecParsSquare(const DDFilteredView&, HGCalParameters&);
  void loadSpecParsHexagon(const DDFilteredView&, HGCalParameters&,
			   const DDCompactView*, const std::string&, 
			   const std::string&);

private:
  std::vector<double> getDDDArray(const std::string&, const DDsvalues_type&,
				  int&);
  std::pair<double,double> cellPosition(const std::map<int,GlobalPoint>& wafers,
					std::map<int,GlobalPoint>::const_iterator& itrf,
					unsigned int num, double rmax,
					double ymax, double xx, double yy, 
					unsigned int ncells);

  struct layerParameters {
    double rmin, rmax, zpos;
    layerParameters(double rin=0, double rout=0, 
		    double zp=0) : rmin(rin), rmax(rout), zpos(zp) {}
  };
  struct cellParameters {
    bool        half;
    GlobalPoint xyz;
    cellParameters(bool h=false, 
		   GlobalPoint p=GlobalPoint(0,0,0)) : half(h), xyz(p) {}
  };

  double                waferSize_;
};

#endif
