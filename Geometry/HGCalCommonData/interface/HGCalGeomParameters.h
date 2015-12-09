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

#include<string>
#include<vector>
#include<iostream>

#include "DetectorDescription/Core/interface/DDsvalues.h"
class DDCompactView;    
class DDFilteredView;
class HGCalParameters;

class HGCalGeomParameters {

public:

  HGCalGeomParameters();
  ~HGCalGeomParameters();
  void loadGeometrySquare(const DDFilteredView&, HGCalParameters&,
			  const std::string&);
  void loadSpecParsSquare(const DDFilteredView&, HGCalParameters&);

private:
  std::vector<double> getDDDArray(const std::string&, const DDsvalues_type&,
				  int&);
};

#endif
