#include <DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDValue.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <DetectorDescription/Core/interface/DDExpandedView.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/OfflineDBLoader/interface/ReadWriteORA.h"

#include "DataSvc/RefException.h"
#include "CoralBase/Exception.h"

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
				  , const DDCompactView& cpv ) {

   try {
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
	std::ofstream dump("dumpGeoHistory");
	do {
	  nav_type pos = epv.navPos();
	  idMap[pos]=id;
	  dump << id << " - " << epv.geoHistory();
	  if ( dumpPosInfo ) {
	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.translation().x();
	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.translation().y();
	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.translation().z();
            dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().thetaX()/deg;
	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().phiX()/deg;
	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().thetaY()/deg;
            dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().phiY()/deg;
	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().thetaZ()/deg;
	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().phiZ()/deg;
//             dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().xx();
// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().xy();
// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().xz();
//             dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().yx();
// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().yy();
// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().yz();
// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().zx();
// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().zy();
// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(5) << epv.rotation().zz();
	  }
	  dump << std::endl;;
	  ++id;
	} while (epv.next());
	dump.close();
      }
      if ( dumpSpecs ) {
	DDSpecifics::iterator<DDSpecifics> spit(DDSpecifics::begin()), spend(DDSpecifics::end());
	// ======= For each DDSpecific...
	std::ofstream dump("dumpSpecs");
	for (; spit != spend; ++spit) {
	  if ( !spit->isDefined().second ) continue;  
	  const DDSpecifics & sp = *spit;
	  dump << sp << std::endl;
	}
	dump.close();
      }
   }catch ( const DDLogicalPart& iException ){  // does this make any sense?
      throw cms::Exception("Geometry")
	<<"GeometryInfoDump::dumpInfo caught a DDLogicalPart exception: \""<<iException<<"\"";
   } catch (const coral::Exception& e) {
      throw cms::Exception("Geometry")
	<<"GeometryInfoDump::dumpInfo caught coral::Exception: \""<<e.what()<<"\"";
   } catch( const pool::RefException& er){
      throw cms::Exception("Geometry")
	<<"GeometryInfoDump::dumpInfo caught pool::RefException: \""<<er.what()<<"\"";
   } catch ( pool::Exception& e ) {
     throw cms::Exception("Geometry")
       << "GeometryInfoDump::dumpInfo caught pool::Exception: \"" << e.what() << "\"";
   } catch ( std::exception& e ) {
     throw cms::Exception("Geometry")
       <<  "GeometryInfoDump::dumpInfo caught std::exception: \"" << e.what() << "\"";
   } catch ( ... ) {
     throw cms::Exception("Geometry")
       <<  "GeometryInfoDump::dumpInfo caught UNKNOWN!!! exception." << std::endl;
   }
}

