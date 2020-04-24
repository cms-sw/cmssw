#ifndef HGCalCommonData_HGCalGeomParameters_h
#define HGCalCommonData_HGCalGeomParameters_h

/** \class HGCalGeomParameters
 *
 * this class extracts some geometry constants from CompactView
 * to be used by Reco Geometry/Topology
 *  
 *  $Date: 2015/06/25 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
 * \author Lindsey Gray, Fermilab <lagray@fnal.gov> (for fixes)
 *
 */

#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

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
			   const std::string&, const std::string&, 
			   HGCalGeometryMode::WaferMode);
  void loadSpecParsSquare(const DDFilteredView&, HGCalParameters&);
  void loadSpecParsHexagon(const DDFilteredView&, HGCalParameters&,
			   const DDCompactView*, const std::string&, 
			   const std::string&);
  void loadWaferHexagon(HGCalParameters& php);
  void loadCellParsHexagon(const DDCompactView* cpv, HGCalParameters& php);

private:

  struct layerParameters {
    double rmin, rmax, zpos;
    layerParameters(double rin=0, double rout=0, 
		    double zp=0) : rmin(rin), rmax(rout), zpos(zp) {}
  };
  struct cellParameters {
    bool        half;
    int         wafer;
    GlobalPoint xyz;
    cellParameters(bool h=false, int w=0, 
		   GlobalPoint p=GlobalPoint(0,0,0)) : half(h), wafer(w), 
      xyz(std::move(p)) {}
  };

  std::vector<double> getDDDArray(const std::string&, const DDsvalues_type&,
				  int&);
  std::pair<double,double> cellPosition(const std::vector<cellParameters>& wafers,
					std::vector<cellParameters>::const_iterator& itrf,
					int wafer, double xx, double yy);

  double                waferSize_;
};

#endif
