#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDetIdBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


CmsTrackerDetIdBuilder::CmsTrackerDetIdBuilder(){
  // Read the file with the map between detid and navtype to restore backward compatibility between 12* and 13* series
  std::cout << " **************************************************************** " << std::endl;
  std::cout << "       You are running Tracker numbering scheme with rr patch     " << std::endl;
  std::cout << "          backward compatibility with CMSSW_1_2_0 restored        " << std::endl;
  std::cout << " **************************************************************** " << std::endl;
  std::string theNavTypeToDetIdMap_FileName = edm::FileInPath("Geometry/TrackerNumberingBuilder/data/ModuleNumbering_120.dat").fullPath();
  std::cout <<" ECCO "<<theNavTypeToDetIdMap_FileName<<std::endl;
  std::ifstream theNavTypeToDetIdMap_File(theNavTypeToDetIdMap_FileName.c_str());
  if (!theNavTypeToDetIdMap_File) 
    cms::Exception("LogicError") <<" File not found "<<theNavTypeToDetIdMap_FileName;
  // fill the map
  uint32_t detid;
  detid = 0;
  std::string navType;
  float x,y,z;
  x=y=z=0;
  std::vector<unsigned int> navtype;
  //
  while(theNavTypeToDetIdMap_File) {
    //
    theNavTypeToDetIdMap_File >> detid
			      >> navType
			      >> x >> y >> z;    
    //
    navtype.clear();
    mapNavTypeToDetId[navType] = detid;
  }
  //
}

GeometricDet* CmsTrackerDetIdBuilder::buildId(GeometricDet* tracker){

  DetId t;
  t = DetId(DetId::Tracker,0);
  tracker->setGeographicalID(t);
  iterate(tracker,0,tracker->geographicalID().rawId());

  return tracker;
}

void CmsTrackerDetIdBuilder::iterate(GeometricDet* in, int level, unsigned int ID){
  if(level == 0){

    for(uint32_t i=0; i<(in)->components().size();i++){
      uint32_t temp1 = ((in)->components())[i]->geographicalID().rawId();
      uint32_t temp = ID;
      temp |= (temp1<<25);
      ((in)->components())[i]->setGeographicalID(DetId(temp));	
      if((temp1==2&&((in)->components())[i]->translation().z()<0.)||
	 (temp1==4&&(((in)->components())[i])->components()[0]->translation().z()<0.)||
	 (temp1==6&&((in)->components())[i]->translation().z()<0.)){
	temp|= (1<<23);
	((in)->components())[i]->setGeographicalID(DetId(temp));	
      }
      
      if((temp1==2&&((in)->components())[i]->translation().z()>0.)||
	 (temp1==4&&(((in)->components())[i])->components()[0]->translation().z()>0.)||
	 (temp1==6&&((in)->components())[i]->translation().z()>0.)){
	temp|= (2<<23);
	((in)->components())[i]->setGeographicalID(DetId(temp));	
      }
      
    }
    
    for (uint32_t k=0; k<(in)->components().size(); k++){
      iterate(((in)->components())[k],level+1,((in)->components())[k]->geographicalID().rawId());
    }
    
  }else if(level==1){
    
    for (uint32_t i=0;i<(in)->components().size();i++){
      uint32_t temp = ID;
      temp |= (((in)->components())[i]->geographicalID().rawId()<<16);
      ((in)->components())[i]->setGeographicalID(DetId(temp));
    }
    
    for (uint32_t k=0; k<(in)->components().size(); k++){
      iterate(((in)->components())[k],level+1,((in)->components())[k]->geographicalID().rawId());
    }

  }else if(level==2){

    for (uint32_t i=0;i<(in)->components().size();i++){
      uint32_t temp = ID;
      temp |= (((in)->components())[i]->geographicalID().rawId()<<8);
      ((in)->components())[i]->setGeographicalID(DetId(temp));
    }

    for (uint32_t k=0; k<(in)->components().size(); k++){
      iterate(((in)->components())[k],level+1,((in)->components())[k]->geographicalID().rawId());
    }
    
  }else if(level==3){
    uint32_t mask = (7<<25);
    uint32_t k = ID & mask;
    k = k >> 25 ;

    if(k==6){
      for (uint32_t i=0;i<(in)->components().size();i++){
	uint32_t temp = ID;
	temp |= (((in)->components())[i]->geographicalID().rawId()<<5);
	((in)->components())[i]->setGeographicalID(DetId(temp));
      }
      
      for (uint32_t k=0; k<(in)->components().size(); k++){
	iterate(((in)->components())[k],level+1,((in)->components())[k]->geographicalID().rawId());
      }
            
    }else{

      for (uint32_t i=0;i<(in)->components().size();i++){
	uint32_t temp = ID;
	temp |= (((in)->components())[i]->geographicalID().rawId()<<2);
	((in)->components())[i]->setGeographicalID(DetId(temp));
      }
      
      for (uint32_t k=0; k<(in)->components().size(); k++){
	iterate(((in)->components())[k],level+1,((in)->components())[k]->geographicalID().rawId());
      }
      
    }
    
  }else if(level==4){
    uint32_t mask = (7<<25);
    uint32_t k = ID & mask;
    k = k >> 25 ;
    if(k==6){
      for (uint32_t i=0;i<(in)->components().size();i++){
	uint32_t temp = ID;
	temp |= (((in)->components())[i]->geographicalID().rawId()<<2);
	((in)->components())[i]->setGeographicalID(DetId(temp));
      }
      
      for (uint32_t k=0; k<(in)->components().size(); k++){
	iterate(((in)->components())[k],level+1,((in)->components())[k]->geographicalID().rawId());
      }
    }else{    
      for (uint32_t i=0;i<(in)->components().size();i++){
	uint32_t temp = ID;
	temp |= (((in)->components())[i]->geographicalID().rawId());
	((in)->components())[i]->setGeographicalID(DetId(temp));
      }
      
    }
  }else if(level==5){
      for (uint32_t i=0;i<(in)->components().size();i++){
	uint32_t temp = ID;
	temp |= (((in)->components())[i]->geographicalID().rawId());
	((in)->components())[i]->setGeographicalID(DetId(temp));
      }

  }

  // Restore compatibility between 12* and 13* series using the map
  std::vector<unsigned int> detNavTypeVector;
  for (uint32_t i=0;i<(in)->components().size();i++){
    GeometricDet::nav_type detNavType = ((in)->components())[i]->navType();
    std::string stringNavType;
    std::stringstream InputOutput(std::stringstream::in | std::stringstream::out);//"tmp.log",std::ios::out);
    // stringstream
    InputOutput << detNavType;
    InputOutput >> stringNavType;
    //
    if( mapNavTypeToDetId[stringNavType] != 0 // to replace only the ones present in the map
	&&
	mapNavTypeToDetId[stringNavType] != ((in)->components())[i]->geographicalID().rawId() ) { // to replace only the ones with detid different wrt 120 map
      std::cout << "\tnavtype " << stringNavType << " detid from map " << mapNavTypeToDetId[stringNavType]
		<< " from det " << ((in)->components())[i]->geographicalID().rawId() << std::endl;
      std::cout << "\t\t replacing " << ((in)->components())[i]->geographicalID().rawId()
		<< " with " << mapNavTypeToDetId[stringNavType] << std::endl;
      ((in)->components())[i]->setGeographicalID(DetId(mapNavTypeToDetId[stringNavType]));
    }
    //
  }
  //
  
  return;

}

