// Last commit: $Id: testSiStripQuality.cc,v 1.0 2013/03/07 10:25:42 querten Exp $

#include "CalibFormats/SiStripObjects/test/plugins/testSiStripQuality.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
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

void testSiStripQuality::analyze( const edm::Event& event, const edm::EventSetup& setup ) {
   edm::ESHandle<SiStripQuality> SiStripQuality_;
   setup.get<SiStripQualityRcd>().get(SiStripQuality_);

   std::vector<SiStripQuality::BadComponent> badComponents = SiStripQuality_->getBadComponentList();
   printf("Size of badComponent = %lu\n", badComponents.size());
   for(unsigned int b=0;b<badComponents.size();b++){
      //printf("% 8i %1i %1i %1i\n", badComponents[b].detid,  (int)badComponents[b].BadModule, badComponents[b].BadFibers, badComponents[b].BadApvs);
      printf("DetId:% 8i BadAPV %1i %1i %1i %1i %1i %1i\n", badComponents[b].detid,  (badComponents[b].BadApvs&1)==1, (badComponents[b].BadApvs&2)==2, (badComponents[b].BadApvs&4)==4, (badComponents[b].BadApvs&8)==8, (badComponents[b].BadApvs&16)==16, (badComponents[b].BadApvs&32)==32);
   }


}


