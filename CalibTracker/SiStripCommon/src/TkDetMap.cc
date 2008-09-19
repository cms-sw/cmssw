#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "iostream"

TkLayerMap::TkLayerMap(int in){

  LogTrace("TkLayerMap") <<"TkLayerMap::constructor ";

  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();

  std::vector<uint32_t> TkDetIdList=fr->getAllDetIds();
  std::vector<uint32_t> LayerDetIdList;
  LogTrace("TkLayerMap") << TkDetIdList.size() << " reduced size " << LayerDetIdList.size();


  SiStripSubStructure siStripSubStructure;
  siStripSubStructure.getTOBDetectors(TkDetIdList,LayerDetIdList,3,0,1);

  LogTrace("TkLayerMap") << TkDetIdList.size() << " reduced size " << LayerDetIdList.size();

 
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
    
    LogTrace("TkLayerMap") << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;

    XYbin xybin2 =   DetToBin[LayerDetIdList[i]];
    LogTrace("TkLayerMap") << xybin2.ix << " " << xybin2.iy  << " " << xybin2.x << " " << xybin2.y ;
  }

/*
  switch (in)
    {
    case TkLayerMap::TIB_L1:

    case TkLayerMap::TIB_L2:

    case TkLayerMap::TIB_L3:

    case TkLayerMap::TIB_L4:         

    case TkLayerMap::TID_D1:

    case TkLayerMap::TID_D2:

    case TkLayerMap::TID_D3:

    case TkLayerMap::TOB_L1:

    case TkLayerMap::TOB_L2:

    case TkLayerMap::TOB_L3:

    case TkLayerMap::TOB_L4:

    case TkLayerMap::TOB_L5:

    case TkLayerMap::TOB_L6:

    case TkLayerMap::TEC_W1:

    case TkLayerMap::TEC_W2:

    case TkLayerMap::TEC_W3:

    case TkLayerMap::TEC_W4:

    case TkLayerMap::TEC_W5: 

    case TkLayerMap::TEC_W6:

    case TkLayerMap::TEC_W7:

    case TkLayerMap::TEC_W8:

    case TkLayerMap::TEC_W9:

    };
*/
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

  LogTrace("TkDetMap") <<"[getXY] detid "<< detid << " cache " << cached_detid;

  if(detid==cached_detid)
    return cached_XYbin;

  /*FIXME*/
  //if (layer!=INVALID)
  FindLayer(detid);
  LogTrace("TkDetMap") <<"[getXY] detid "<< detid << " cache " << cached_detid << cached_XYbin.ix << " " << cached_XYbin.iy  << " " << cached_XYbin.x << " " << cached_XYbin.y ;    
  return cached_XYbin;
}

TkLayerMap::TkLayerEnum TkDetMap::FindLayer(uint32_t& detid){ 

  if(detid==cached_detid)
    return cached_layer;

  cached_detid=detid;
  cached_layer=TkLayerMap::TOB_L1;  /*FIXME*/ 
  cached_iterator=TkMap.find(cached_layer);
  cached_XYbin=cached_iterator->second->getXY(detid);
  return cached_layer;
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
