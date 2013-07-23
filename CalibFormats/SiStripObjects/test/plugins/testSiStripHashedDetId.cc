// Last commit: $Id: testSiStripHashedDetId.cc,v 1.2 2010/01/07 10:25:42 demattia Exp $

#include "CalibFormats/SiStripObjects/test/plugins/testSiStripHashedDetId.h"
#include "CalibFormats/SiStripObjects/interface/SiStripHashedDetId.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include <boost/cstdint.hpp>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <time.h>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
testSiStripHashedDetId::testSiStripHashedDetId( const edm::ParameterSet& pset ) 
{
  edm::LogVerbatim(mlDqmCommon_)
    << "[testSiStripHashedDetId::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
testSiStripHashedDetId::~testSiStripHashedDetId() {
  edm::LogVerbatim(mlDqmCommon_)
    << "[testSiStripHashedDetId::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void testSiStripHashedDetId::initialize( const edm::EventSetup& setup ) {
  edm::LogVerbatim(mlDqmCommon_)
    << "[SiStripHashedDetId::" << __func__ << "]"
    << " Tests the generation of DetId hash map...";
  
  // Retrieve geometry
  edm::ESHandle<TrackerGeometry> geom;
  setup.get<TrackerDigiGeometryRecord>().get( geom );

  // Build list of DetIds
  std::vector<uint32_t> dets;
  dets.reserve(16000);
  TrackerGeometry::DetUnitContainer::const_iterator iter = geom->detUnits().begin();
  for( ; iter != geom->detUnits().end(); ++iter ) {
    const StripGeomDetUnit* strip = dynamic_cast<StripGeomDetUnit*>(*iter);
    if( strip ) {
      dets.push_back( (strip->geographicalId()).rawId() );
    }
  }
  edm::LogVerbatim(mlDqmCommon_)
    << "[testSiStripHashedDetId::" << __func__ << "]"
    << " Retrieved " << dets.size() 
    << " strip DetIds from geometry!";

  // Sorted DetId list gives max performance, anything else is worse
  if ( true ) { std::sort( dets.begin(), dets.end() ); }
  else { std::reverse( dets.begin(), dets.end() ); }
    
  // Manipulate DetId list
  if ( false ) {
    if ( dets.size() > 4 ) {
      uint32_t temp = dets.front();
      dets.front() = dets.back();          // swapped
      dets.back() = temp;                  // swapped
      dets.at(1) = 0x00000001;             // wrong
      dets.at(dets.size()-2) = 0xFFFFFFFF; // wrong
    }
  }

  // Create hash map
  SiStripHashedDetId hash( dets );
  LogTrace(mlDqmCommon_)
    << "[testSiStripHashedDetId::" << __func__ << "]"
    << " DetId hash map: " << std::endl
    << hash;

  // Manipulate DetId list
  if ( false ) {
    if ( dets.size() > 4 ) {
      uint32_t temp = dets.front();
      dets.front() = dets.back();          // swapped
      dets.back() = temp;                  // swapped
      dets.at(1) = 0x00000001;             // wrong
      dets.at(dets.size()-2) = 0xFFFFFFFF; // wrong
    }
  }
  
  // Retrieve hashed indices
  std::vector<uint32_t> hashes;
  uint32_t istart = time(NULL);
  for( uint16_t tt = 0; tt < 10000; ++tt ) { // 10000 loops just to see some non-negligible time meaasurement!
    hashes.clear();
    hashes.reserve(dets.size());
    std::vector<uint32_t>::const_iterator idet = dets.begin();
    for( ; idet != dets.end(); ++idet ) {
      hashes.push_back( hash.hashedIndex(*idet) );
    }
  }
  
  // Some debug
  std::stringstream ss;
  ss << "[testSiStripHashedDetId::" << __func__ << "]";
  std::vector<uint32_t>::const_iterator ii = hashes.begin();
  uint16_t cntr1 = 0;
  for( ; ii != hashes.end(); ++ii ) {
    if ( *ii == sistrip::invalid32_ ) {
      cntr1++;
      ss << std::endl
	 << " Invalid index " 
	 << *ii;
      continue;
    }
    uint32_t detid = hash.unhashIndex(*ii);
    std::vector<uint32_t>::const_iterator iter = find( dets.begin(), dets.end(), detid );
    if ( iter == dets.end() ) {
      cntr1++;
      ss << std::endl
	 << " Did not find value " 
	 << detid << " at index " 
	 << ii-hashes.begin()
	 << " in vector!";
    } else if ( *ii != static_cast<uint32_t>(iter-dets.begin()) ) {
      cntr1++;
      ss << std::endl
	 << " Found same value " 
	 << detid << " at different indices " 
	 << *ii << " and " 
	 << iter-dets.begin();
    }
  }
  if ( cntr1 ) { ss << std::endl << " Found " << cntr1 << " incompatible values!"; }
  else { ss << " Found no incompatible values!"; }
  LogTrace(mlDqmCommon_) << ss.str();

  edm::LogVerbatim(mlDqmCommon_)
    << "[testSiStripHashedDetId::" << __func__ << "]"
    << " Processed " << hashes.size()
    << " DetIds in " << (time(NULL)-istart)
    << " seconds";
  
  // Retrieve DetIds
  std::vector<uint32_t> detids;
  uint32_t jstart = time(NULL);
  for( uint16_t ttt = 0; ttt < 10000; ++ttt ) { // 10000 loops just to see some non-negligible time meaasurement!
    detids.clear();
    detids.reserve(dets.size());
    for( uint16_t idet = 0; idet < dets.size(); ++idet ) {
      detids.push_back( hash.unhashIndex(idet) );
    }
  }
  
  // Some debug
  std::stringstream sss;
  sss << "[testSiStripHashedDetId::" << __func__ << "]";
  uint16_t cntr2 = 0;
  std::vector<uint32_t>::const_iterator iii = detids.begin();
  for( ; iii != detids.end(); ++iii ) {
    if ( *iii != dets.at(iii-detids.begin()) ) {
      cntr2++;
      sss << std::endl
	  << " Diff values " 
	  << *iii << " and " 
	  << dets.at(iii-detids.begin())
	  << " found at index " 
	  << iii-detids.begin() << " ";
    }
  }
  if ( cntr2 ) { sss << std::endl << " Found " << cntr2 << " incompatible values!"; }
  else { sss << " Found no incompatible values!"; }
  LogTrace(mlDqmCommon_) << sss.str();

  edm::LogVerbatim(mlDqmCommon_)
    << "[testSiStripHashedDetId::" << __func__ << "]"
    << " Processed " << detids.size()
    << " hashed indices in " << (time(NULL)-jstart)
    << " seconds";
 
}

// -----------------------------------------------------------------------------
// 
void testSiStripHashedDetId::analyze( const edm::Event& event, 
				       const edm::EventSetup& setup ) {
  initialize(setup);
  LogTrace(mlDqmCommon_) 
    << "[testSiStripHashedDetId::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}


