#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "iostream"

TkLayerMap::TkLayerMap(int in){

  LogTrace("TkLayerMap") <<" TkLayerMap::constructor for layer " << in;

  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();

  std::vector<uint32_t> TkDetIdList=fr->getAllDetIds();
   
  switch (in)
    {
    case TkLayerMap::TIB_L1:
      createTIB12(TkDetIdList,1);
      break;
    case TkLayerMap::TIB_L2:
      createTIB12(TkDetIdList,2);
      break;
    case TkLayerMap::TIB_L3:
      createTIB34(TkDetIdList,3);
      break;
    case TkLayerMap::TIB_L4:         
      createTIB34(TkDetIdList,4);
      break;
    case TkLayerMap::TID_D1:
      createTID13(TkDetIdList,1); 
      break;
    case TkLayerMap::TID_D2:
      createTID13(TkDetIdList,2); 
      break;
    case TkLayerMap::TID_D3:
      createTID13(TkDetIdList,3); 
      break;
    case TkLayerMap::TOB_L1:
      createTOB12(TkDetIdList,1); 
      break;
    case TkLayerMap::TOB_L2:
      createTOB12(TkDetIdList,2); 
      break;
    case TkLayerMap::TOB_L3:
      createTOB36(TkDetIdList,3); 
      break;
    case TkLayerMap::TOB_L4:
      createTOB36(TkDetIdList,4); 
      break;
    case TkLayerMap::TOB_L5:
      createTOB36(TkDetIdList,5); 
      break;
    case TkLayerMap::TOB_L6:
      createTOB36(TkDetIdList,6); 
      break;
    case TkLayerMap::TEC_W1:
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W2:
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W3:
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W4:
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W5: 
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W6:
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W7:
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W8:
      createTest(TkDetIdList); 
      break;
    case TkLayerMap::TEC_W9:
      createTest(TkDetIdList); 
      break;
    };
}

void TkLayerMap::createTOB36(std::vector<uint32_t>& TkDetIdList,int layer){

  int Nrod;
  switch (layer){
  case 3:
    Nrod=54;
    break;
  case 4:
    Nrod=60;
    break;
  case 5:
    Nrod=66;
    break;
  case 6:
    Nrod=74;
    break;
  }

  nchX=12;
  lowX=-6;
  highX=6;
  nchY=Nrod;
  lowY=0;
  highY=Nrod;

  binToDet.resize(nchX*nchY);

  std::vector<uint32_t> LayerDetIdList, DetIdList;
  SiStripSubStructure siStripSubStructure;

  //extract  vector of module in the layer
  siStripSubStructure.getTOBDetectors(TkDetIdList,LayerDetIdList,layer,0,0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTOB36] layer " << layer  << " number of dets " << LayerDetIdList.size();
  XYbin xybin;

  for(int zmin_plus=1;zmin_plus<3;zmin_plus++) //loop on Z- and Z+
    for(int rod=1;rod<=Nrod;rod++){
      DetIdList.clear();
      siStripSubStructure.getTOBDetectors(LayerDetIdList,DetIdList,layer,0,rod);
      LogTrace("TkLayerMap") << "[TkLayerMap::rod] " << rod <<  "  size " << DetIdList.size();
      for(size_t j=0;j<DetIdList.size();++j){
	if(j/6==0){
	  xybin.ix=6-j;
	  xybin.iy=rod;
	}else{
	  xybin.ix=j+1;
	  xybin.iy=rod;
	}
	xybin.x=lowX+xybin.ix-0.5;
	xybin.y=rod-0.5;
	
	DetToBin[DetIdList[j]]=xybin;
	binToDet[(xybin.ix-1)+nchX*(xybin.iy-1)]=DetIdList[j];
	LogTrace("TkLayerMap") << "[TkLayerMap::createTOB36] " << DetIdList[j]<< " " << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;
      }
    }
}

void TkLayerMap::createTest(std::vector<uint32_t>& TkDetIdList){

  std::vector<uint32_t> LayerDetIdList;
  LogTrace("TkLayerMap") << "[TkLayerMap::createTest] " << TkDetIdList.size() << " reduced size " << LayerDetIdList.size();

  SiStripSubStructure siStripSubStructure;
  siStripSubStructure.getTOBDetectors(TkDetIdList,LayerDetIdList,3,0,1);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTest] " << TkDetIdList.size() << " reduced size " << LayerDetIdList.size();

 
  nchX=3;
  nchY=LayerDetIdList.size();
  lowX=-0.5;
  highX=2.5;
  lowY=-0.5;
  highY=nchY-0.5;

  binToDet.resize(nchX*nchY);
  for(size_t i=0;i<LayerDetIdList.size();++i){
    int theix=1;
    int theiy=i+1;
    float thex=theix-1;
    float they=theiy-1;
    
    LogTrace("TkLayerMap") << LayerDetIdList[i] << std::endl;
    XYbin xybin(theix,theiy,thex,they);
    DetToBin[LayerDetIdList[i]]=xybin;
    binToDet[(theix-1)+nchX*(theiy-1)]=LayerDetIdList[i];
    
    LogTrace("TkLayerMap") << "[TkLayerMap::createTest] "<< xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;

    XYbin xybin2 =   DetToBin[LayerDetIdList[i]];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTest] "<< xybin2.ix << " " << xybin2.iy  << " " << xybin2.x << " " << xybin2.y ;
  }

}


const TkLayerMap::XYbin& TkLayerMap::getXY(uint32_t& detid){
  /*FIXME*/
  return DetToBin[detid];
}

  //--------------------------------------

TkDetMap::TkDetMap(const edm::ParameterSet& p,const edm::ActivityRegistry& a)
  :cached_detid(0),
   cached_layer(TkLayerMap::INVALID){
  doMe();
}

TkDetMap::TkDetMap()
:cached_detid(0),
 cached_layer(TkLayerMap::INVALID){
  doMe();
}

void TkDetMap::doMe(){
  LogTrace("TkDetMap") <<"TkDetMap::constructor ";

  //Create TkLayerMap for each layer declared in the TkLayerEnum
  for(int layer=1;layer<23;++layer){
    TkMap[layer]=new TkLayerMap(layer);
  }
}

TkDetMap::~TkDetMap(){
  detmapType::iterator iter=TkMap.begin();
  detmapType::iterator iterE=TkMap.end();
 
  for(;iter!=iterE;++iter)
    delete iter->second;  
}

const TkLayerMap::XYbin& TkDetMap::getXY(uint32_t& detid){

  LogTrace("TkDetMap") <<"[getXY] detid "<< detid << " cache " << cached_detid << " layer " << cached_layer << " XY " << cached_XYbin.ix << " " << cached_XYbin.iy  << " " << cached_XYbin.x << " " << cached_XYbin.y ;    
  if(detid==cached_detid)
    return cached_XYbin;

  /*FIXME*/
  //if (layer!=INVALID)
  FindLayer(detid);
  LogTrace("TkDetMap") <<"[getXY] detid "<< detid << " cache " << cached_detid << " layer " << cached_layer << " XY " << cached_XYbin.ix << " " << cached_XYbin.iy  << " " << cached_XYbin.x << " " << cached_XYbin.y ;    
  return cached_XYbin;
}

int16_t TkDetMap::FindLayer(uint32_t& detid){ 

  if(detid==cached_detid)
    return cached_layer;

  cached_detid=detid;

  int16_t layer=layerSearch(detid);
  LogTrace("TkDetMap") <<"[FindLayer] detid "<< detid << " layer " << layer;
  if(layer!=cached_layer){
    cached_layer=layer;  
    cached_iterator=TkMap.find(cached_layer);
  }
  cached_XYbin=cached_iterator->second->getXY(detid);

  return cached_layer;
}


int16_t TkDetMap::layerSearch(uint32_t detid){
  switch((detid>>25)&0x7){
  case SiStripDetId::TIB:
    return ((detid>>14)&0x7);
  case SiStripDetId::TID:
    return 4+((detid>>11)&0x3);
  case SiStripDetId::TOB:
    return 7+((detid>>14)&0x7);
  case SiStripDetId::TEC:
    return 13+((detid>>14)&0xF);
  }
  return 0;
}

void TkDetMap::getComponents(int& layer,
			     int& nchX,double& lowX,double& highX,
			     int& nchY,double& lowY,double& highY){
  
   detmapType::const_iterator iter=TkMap.find(layer);
   nchX=iter->second->get_nchX();
   lowX=iter->second->get_lowX();
   highX=iter->second->get_highX();
   nchY=iter->second->get_nchY();
   lowY=iter->second->get_lowY();
   highY=iter->second->get_highY();
}

std::string TkDetMap::getLayerName(int& in){
  switch (in)
    {
    case TkLayerMap::TIB_L1:
      return "TIB_L1";
    case TkLayerMap::TIB_L2:
      return "TIB_L2";
    case TkLayerMap::TIB_L3:
      return "TIB_L3";
    case TkLayerMap::TIB_L4:         
      return "TIB_L4";         
    case TkLayerMap::TID_D1:
      return "TID_D1";
    case TkLayerMap::TID_D2:
      return "TID_D2";
    case TkLayerMap::TID_D3:
      return "TID_D3";
    case TkLayerMap::TOB_L1:
      return "TOB_L1";
    case TkLayerMap::TOB_L2:
      return "TOB_L2";
    case TkLayerMap::TOB_L3:
      return "TOB_L3";
    case TkLayerMap::TOB_L4:
      return "TOB_L4";
    case TkLayerMap::TOB_L5:
      return "TOB_L5";
    case TkLayerMap::TOB_L6:
      return "TOB_L6";
    case TkLayerMap::TEC_W1:
      return "TEC_W1";
    case TkLayerMap::TEC_W2:
      return "TEC_W2";
    case TkLayerMap::TEC_W3:
      return "TEC_W3";
    case TkLayerMap::TEC_W4:
      return "TEC_W4";
    case TkLayerMap::TEC_W5: 
      return "TEC_W5";
    case TkLayerMap::TEC_W6:
      return "TEC_W6";
    case TkLayerMap::TEC_W7:
      return "TEC_W7";
    case TkLayerMap::TEC_W8:
      return "TEC_W8";
    case TkLayerMap::TEC_W9:
      return "TEC_W9";
    }
  return "Invalid";
}
