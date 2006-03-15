#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDetIdBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "DataFormats/DetId/interface/DetId.h"


CmsTrackerDetIdBuilder::CmsTrackerDetIdBuilder(){
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

  return;

}

