#include <DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h>

#include <DetectorDescription/Core/interface/DDValue.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include "DetectorDescription/Core/interface/DDName.h"


#include <iostream>
#include <fstream>

// Need thes??? maybe 
#include <cmath>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>

GeometryInfoDump::GeometryInfoDump () { }

GeometryInfoDump::~GeometryInfoDump () { }
  

void GeometryInfoDump::dumpInfo ( bool dumpHistory, bool dumpSpecs, bool dumpPosInfo
				  , const DDCompactView& cpv, std::string fname, int nVols ) {
  fname = "dump" + fname;
  DDExpandedView epv(cpv);
  std::cout << "Top Most LogicalPart =" << epv.logicalPart() << std::endl;
  if ( dumpHistory || dumpPosInfo) {
    if ( dumpPosInfo ) {
      std::cout << "After the GeoHistory in the output file dumpGeoHistoryOnRead you will see x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33" << std::endl;
    }
    typedef DDExpandedView::nav_type nav_type;
    typedef std::map<nav_type,int> id_type;
    id_type idMap;
    int id=0;
    std::ofstream dump(fname.c_str());
    bool notReachedDepth(true);
    do {
      nav_type pos = epv.navPos();
      idMap[pos]=id;
      //      dump << id 
      dump << " - " << epv.geoHistory();
      DD3Vector x, y, z;
      epv.rotation().GetComponents(x,y,z);
      if ( dumpPosInfo ) {
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.translation().x();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.translation().y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.translation().z();
	//             dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().thetaX()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().phiX()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().thetaY()/deg;
	//             dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().phiY()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().thetaZ()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().phiZ()/deg;
	
	//          dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().xx();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().xy();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().xz();
	//          dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().yx();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().yy();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().yz();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().zx();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().zy();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().zz();
	
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.X();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.X();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.X();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Z();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Z();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Z();
      }
      dump << std::endl;;
      ++id;
      if ( nVols != 0 && id > nVols ) notReachedDepth = false;
    } while (epv.next() && notReachedDepth);
    dump.close();
  }
  if ( dumpSpecs ) {
    DDSpecifics::iterator<DDSpecifics> spit(DDSpecifics::begin()), spend(DDSpecifics::end());
    // ======= For each DDSpecific...
    std::string dsname = "dumpSpecs" + fname;
    std::ofstream dump(dsname.c_str());
    for (; spit != spend; ++spit) {
      if ( !spit->isDefined().second ) continue;  
      const DDSpecifics & sp = *spit;
      dump << sp << std::endl;
    }
    dump.close();
  }
}

