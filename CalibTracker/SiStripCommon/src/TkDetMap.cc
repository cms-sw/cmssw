#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "iostream"

TkLayerMap::TkLayerMap(int in):layerEnumNb(in){

  LogTrace("TkLayerMap") <<" TkLayerMap::constructor for layer " << in;

  initialize(layerEnumNb);

  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();

  std::vector<uint32_t> TkDetIdList=fr->getAllDetIds();
  
  if(layerEnumNb==0)
    edm::LogError("TkLayerMap") <<" TkLayerMap::requested creation of a wrong layer Nb "<< layerEnumNb; 
  else if(layerEnumNb<5)
    createTIB(TkDetIdList,layerEnumNb);
  else if(layerEnumNb<8)
    createTID(TkDetIdList,layerEnumNb); 
  else if(layerEnumNb<14)
    createTOB(TkDetIdList,layerEnumNb); 
  else
    createTEC(TkDetIdList,layerEnumNb); 
}

void TkLayerMap::initialize(int layer){

  switch (layer){
  case 1: //TIBL1
    
    Nstring_ext=30;
    SingleExtString.insert(SingleExtString.begin(),8,0);
    SingleExtString.insert(SingleExtString.begin()+8,7,1);
    SingleExtString.insert(SingleExtString.begin()+15,8,2);
    SingleExtString.insert(SingleExtString.begin()+23,7,3);
    nchX=12;
    lowX=-6;
    highX=6;
    nchY=2*(Nstring_ext+1);
    lowY=-1.*(Nstring_ext+1.);
    highY=(Nstring_ext+1);
    
    break;
    case 2:
    
    Nstring_ext=38;
    SingleExtString.insert(SingleExtString.begin(),10,0);
    SingleExtString.insert(SingleExtString.begin()+10,9,1);
    SingleExtString.insert(SingleExtString.begin()+19,10,2);
    SingleExtString.insert(SingleExtString.begin()+29,9,3);
    nchX=12;
    lowX=-6;
    highX=6;
    nchY=2*(Nstring_ext+1);
    lowY=-1.*(Nstring_ext+1.);
    highY=(Nstring_ext+1);

    break;
  case 3:

    Nstring_ext=46; 
    SingleExtString.insert(SingleExtString.begin(),23,0);
    SingleExtString.insert(SingleExtString.begin()+23,23,1);
    nchX=12;
    lowX=-6;
    highX=6;
    nchY=Nstring_ext;
    lowY=0;
    highY=nchY;

    break;
  case 4:

    Nstring_ext=56;
    SingleExtString.insert(SingleExtString.begin(),14,0);
    SingleExtString.insert(SingleExtString.begin()+14,14,1);
    SingleExtString.insert(SingleExtString.begin()+28,14,2);
    SingleExtString.insert(SingleExtString.begin()+42,14,3);
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nstring_ext;
    lowY=0.;
    highY=nchY;

    break;


  case 5:
  case 6:
  case 7:
    
    nchX=16;
    lowX=-8.;
    highX=8.;
    nchY=40;
    lowY=0.0;
    highY=1.*nchY;

    break;
  case 8:  //TOBL1

    Nrod=42;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=2*(Nrod+1);
    lowY=-1.*(Nrod-1.);
    highY=(Nrod+1.);

    break;
  case 9:
    
    Nrod=48;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=2.*(Nrod+1.);
    lowY=-1.*(Nrod-1.);
    highY=(Nrod+1.);
    
    break;
  case 10: //TOBL3
   
    Nrod=54;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;
    
    break;
  case 11:
    
    Nrod=60;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;

    break;
  case 12:
    
    Nrod=66;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;

    break;
  case 13:
    
    Nrod=74;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;

    break;
    //FIXME add TEC

  }
  
  for (size_t i=0;i<SingleExtString.size();i++)
    LogTrace("TkLayerMap") << "[initialize SingleExtString["<<i<<"] " << SingleExtString[i];

  binToDet.resize(nchX*nchY);
}
 
void TkLayerMap::createTIB(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTIBDetectors(TkDetIdList,LayerDetIdList,layerEnumNb,0,0,0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTIB12] layer " << layerEnumNb  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xybin=getXY_TIB(LayerDetIdList[j],layerEnumNb);
    binToDet[(xybin.ix-1)+nchX*(xybin.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTIB] " << LayerDetIdList[j]<< " " << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;
  }
}

void TkLayerMap::createTOB(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTOBDetectors(TkDetIdList,LayerDetIdList,layerEnumNb-7,0,0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTOB] layer " << layerEnumNb-7  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xybin=getXY_TOB(LayerDetIdList[j],layerEnumNb);
    binToDet[(xybin.ix-1)+nchX*(xybin.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTOB] " << LayerDetIdList[j]<< " " << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;
  }
}

void TkLayerMap::createTID(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTIDDetectors(TkDetIdList,LayerDetIdList,layerEnumNb-4,0,0,0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTID] layer " << layerEnumNb-4  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xybin=getXY_TID(LayerDetIdList[j],layerEnumNb);
    binToDet[(xybin.ix-1)+nchX*(xybin.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTID] " << LayerDetIdList[j]<< " " << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;
  }
}

void TkLayerMap::createTEC(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTECDetectors(TkDetIdList,LayerDetIdList,layerEnumNb-13,0,0,0);
  
  LogTrace("TkLayerMap") << "[TkLayerMap::createTOB] layer " << layerEnumNb-13  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xybin=getXY_TEC(LayerDetIdList[j],layerEnumNb);
    binToDet[(xybin.ix-1)+nchX*(xybin.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTID] " << LayerDetIdList[j]<< " " << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;
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


const TkLayerMap::XYbin TkLayerMap::getXY(uint32_t& detid, int layerEnumNb){
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);
  if(layerEnumNb<5)  
    getXY_TIB(detid,layerEnumNb);
  else if(layerEnumNb<8)  
    getXY_TID(detid,layerEnumNb); 
  else if(layerEnumNb<14)  
    getXY_TOB(detid,layerEnumNb); 
  else 
    getXY_TEC(detid,layerEnumNb); 
}

TkLayerMap::XYbin TkLayerMap::getXY_TIB(uint32_t& detid, int layerEnumNb){  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TIBDetId D(detid);
  if(D.layerNumber()%2){
    Offset=D.isInternalString()?2:1;
  }else{
    Offset=D.isInternalString()?1:2;
  }
  xybin.ix=2*(D.isZMinusSide()?-1*D.moduleNumber()+3:D.moduleNumber()+2)+Offset;
  xybin.iy=D.isInternalString()?D.stringNumber()+SingleExtString[D.stringNumber()]:D.stringNumber();
  if(D.layerNumber()<3 && !D.isStereo())
    xybin.iy+=Nstring_ext+2;
  
  xybin.x=lowX+xybin.ix-0.5;
  xybin.y=lowY+xybin.iy-0.5;
  return xybin;
}

TkLayerMap::XYbin TkLayerMap::getXY_TOB(uint32_t& detid, int layerEnumNb){  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TOBDetId D(detid);
  xybin.ix=D.isZMinusSide()?-1*D.moduleNumber()+7:D.moduleNumber()+6;
  xybin.iy=D.rodNumber();  
  if(D.layerNumber()<3 && !D.isStereo())
    xybin.iy+=Nrod+2;

  xybin.x=lowX+xybin.ix-0.5;
  xybin.y=lowY+xybin.iy-0.5;
  return xybin;
}

TkLayerMap::XYbin TkLayerMap::getXY_TID(uint32_t& detid, int layerEnumNb){  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TIDDetId D(detid);
  xybin.ix=D.isZMinusSide()?-3*D.ring()+10:3*D.ring()+7;
  if(D.isStereo())
    xybin.ix+=(D.isZMinusSide()?-1:1);
  xybin.iy=(D.isBackRing()?2:1) * D.moduleNumber()+1;

  xybin.x=lowX+xybin.ix-0.5;
  xybin.y=lowY+xybin.iy-0.5;
  return xybin;
}

TkLayerMap::XYbin TkLayerMap::getXY_TEC(uint32_t& detid, int layerEnumNb){  
  /*FIXME*/
  XYbin pippo;
  return  pippo;
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

  int16_t layer=TkLayerMap::layerSearch(detid);
  LogTrace("TkDetMap") <<"[FindLayer] detid "<< detid << " layer " << layer;
  if(layer!=cached_layer){
    cached_layer=layer;  
    cached_iterator=TkMap.find(cached_layer);
  }
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
