#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CmsTrackerDebugNavigator::CmsTrackerDebugNavigator ( const std::vector<GeometricDetExtra> * gdes ) {
  std::vector<GeometricDetExtra>::const_iterator gdeiEnd(gdes->end());
  std::vector<GeometricDetExtra>::const_iterator gdei(gdes->begin());
   for (; gdei != gdeiEnd; ++gdei) {
     const GeometricDetExtra* gdeaddr (&(*gdei));
     std::cout <<"pointer to GeometricDetExtra " << gdeaddr << " geographicalId() = " << gdei->geographicalId().rawId() << std::endl;
     _helperMap[gdei->geographicalId().rawId()] = gdeaddr;
  }
}

void CmsTrackerDebugNavigator::dump(const GeometricDet* in, const std::vector<GeometricDetExtra> * gdes){
  edm::LogInfo("CmsTrackerDebugNavigator")<<" CmsTrackerDebugNavigator - Starting debug";
  for (int k=0; k<20; k++) numinstances[k]=0;
  iterate(in,0, gdes);
  
  for (int k=0; k<20; k++){
    edm::LogInfo("CmsTrackerDebugNavigator")<<" CmsTrackerDebugNavigator has found "<<numinstances[k]<<" GeometricDets at level "<<k;
  }
}

void CmsTrackerDebugNavigator::iterate(const GeometricDet*in, int level, const std::vector<GeometricDetExtra> * gdes){
  //  static CmsTrackerStringToEnum enumName;
  numinstances[level]++;
//   std::vector<GeometricDetExtra>::const_iterator gdeiEnd(gdes->end());
//   std::vector<GeometricDetExtra>::const_iterator gdei(gdes->begin());
  //    if (level == 2) {
  //      for (;gdei != gdeiEnd; ++gdei) {
  //        std::cout <<"gdei->geographicalId().rawId() = " << gdei->geographicalId().rawId() << std::endl;
  //      }
  //      exit(0);
  //    }
  // what if I want to run once through the GeometricDetExtra's and index them... 
  //  std::map<int, std::vector<GeometricDetExtra>::const_iterator > helperMap; // run makes the next loop faster
  //  std::map<int, const GeometricDetExtra*> helperMap; // run makes the next loop faster
  //   for (; gdei != gdeiEnd; ++gdei) {
  //     //    std::cout << "geographicalId() is " << std::endl;
  //     helperMap[gdei->geographicalId()] = &(*gdei);
  //     //    helperMap[gdei->geographicalId()] = gdei;
  //   }
  //  std::cout << "got to iterate how big is gdes? " <<gdes->size()<< " how big is the map? " << _helperMap.size() << std::endl; 
  std::vector<GeometricDetExtra>::const_iterator gdei(gdes->begin()), gdeEnd(gdes->end());
  for (unsigned int k=0; k<(in)->components().size(); k++){
    std::string spaces = "";
    gdei = gdes->begin();
    for (; gdei != gdeEnd; ++gdei) {
      //      std::cout << " gdei->geographicalId() = " << gdei->geographicalId() << " =? " << (in)->components()[k]->geographicalId() << std::endl; 
      if ( gdei->geographicalId().rawId() == (in)->components()[k]->geographicalId().rawId() ) break;
    }

    //    if (gdei == gdeEnd) throw ("THERE IS NO MATCHING DetId in the GeometricDetExtra"); //THIS never happens!
    //    if (gdei != gdeEnd) 
    std::cout << "CmsMatch " << (in)->components()[k]->geographicalID().rawId() << " in gdes " << gdei->geographicalId().rawId() << std::endl;

    for(unsigned int i=0; (int)i<=level; i++) spaces+="  ";
    //    This search is slow.  Is a map<DetId*, GeometricDetExtra*> worth it?
//     gdei = gdes->begin();
//     for ( ; gdei != gdeiEnd ; ++gdei ) {
//       // over checking? wrong? FIX  && gdei->geometricDet() == in
//       if ( gdei->geographicalId().rawId() == (in)->geographicalId().rawId() ) break;
//     }
//     //  NEW search
//     if (gdei != gdeiEnd) std::cout << " gdei is " << gdei->geographicalId().rawId() << " and address of geometricDet is " << gdei->geometricDet() << std::endl;
//     else std::cout << "not found conventionally " << (in)->geographicalId().rawId() << " " << gdei->geographicalId().rawId() << std::endl;
    const GeometricDetExtra* extra = _helperMap[(in)->geographicalId()];
//     if (extra == 0) {
//       std::cout << " extra " << extra << " for " << (in)->geographicalId() << std::endl;
//     }else{
//       std::cout << "(in)->geographicalId() = " << (in)->geographicalId() << " THIS pointer IS: " << extra << std::endl;
      edm::LogInfo("CmsTrackerDebugNavigator") << level << spaces
					       << "### VOLUME " << (in)->components()[k]->name().name()
					       << " of type " << (in)->components()[k]->type()
	// 					     << " copy number " << (in)->components()[k]->copyno()
					       << " copy number " << extra->copyno()
					       << " positioned in " << (in)->name().name()
					       << " global position of centre " << (in)->components()[k]->translation()
	//	      << " rotation matrix " << (in)->components()[k]->rotation()
					       << " volume = "  << extra->volume()  << " cm3"
 					       << " density = " << extra->density() << " g/cm3"
 					       << " weight "    << extra->weight()  << " g"
	// 					     << " volume = "  << (in)->components()[k]->volume()  << " cm3"
	// 					     << " density = " << (in)->components()[k]->density() << " g/cm3"
	// 					     << " weight "    << (in)->components()[k]->weight()  << " g"
					       << std::endl;
//     }
    iterate(((in)->components())[k],level+1, gdes);
  }
  return;
}
