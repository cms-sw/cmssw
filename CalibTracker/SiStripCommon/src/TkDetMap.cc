#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "iostream"

TkLayerMap::TkLayerMap(int in):layerEnumNb_(in){

  LogTrace("TkLayerMap") <<" TkLayerMap::constructor for layer " << in;

  initialize(layerEnumNb_);

 if(!edm::Service<SiStripDetInfoFileReader>().isAvailable()){
    edm::LogError("TkLayerMap") << 
      "\n------------------------------------------"
      "\nUnAvailable Service SiStripDetInfoFileReader: please insert in the configuration file an instance like"
      "\n\tprocess.SiStripDetInfoFileReader = cms.Service(\"SiStripDetInfoFileReader\")"
      "\n------------------------------------------";
  }
 
  SiStripDetInfoFileReader * fr=edm::Service<SiStripDetInfoFileReader>().operator->();

  std::vector<uint32_t> TkDetIdList=fr->getAllDetIds();
  
  if(layerEnumNb_==0)
    edm::LogError("TkLayerMap") <<" TkLayerMap::requested creation of a wrong layer Nb "<< layerEnumNb_; 
  else if(layerEnumNb_<5)
    createTIB(TkDetIdList,layerEnumNb_);
  else if(layerEnumNb_<8)
    createTID(TkDetIdList,layerEnumNb_); 
  else if(layerEnumNb_<14)
    createTOB(TkDetIdList,layerEnumNb_); 
  else
    createTEC(TkDetIdList,layerEnumNb_); 
}

uint32_t TkLayerMap::getDetFromBin(int ix, int iy){
  
  int val=(ix-1)+nchX*(iy-1);
  if(val>-1 && val < nchX*nchY)
    return binToDet[val];
  return 0;
}

const int16_t TkLayerMap::layerSearch(uint32_t detid){
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
  case 5:  //TID
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
    lowY=-1.*(Nrod+1.);
    highY=(Nrod+1.);

    break;
  case 9:
    
    Nrod=48;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=int(2.*(Nrod+1.));
    lowY=-1.*(Nrod+1.);
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
  default: //TEC
    nchX=34;
    lowX=-17.;
    highX=17.;
    nchY=80;
    lowY=0.;
    highY=1.*nchY;

    ModulesInRingFront.push_back(0); //null value for component 0
    ModulesInRingFront.push_back(2);
    ModulesInRingFront.push_back(2);
    ModulesInRingFront.push_back(3);
    ModulesInRingFront.push_back(4);
    ModulesInRingFront.push_back(2);
    ModulesInRingFront.push_back(4);
    ModulesInRingFront.push_back(5);

    ModulesInRingBack.push_back(0); //null value for component 0
    ModulesInRingBack.push_back(1);    
    ModulesInRingBack.push_back(1);    
    ModulesInRingBack.push_back(2);    
    ModulesInRingBack.push_back(3);    
    ModulesInRingBack.push_back(3);    
    ModulesInRingBack.push_back(3);    
    ModulesInRingBack.push_back(5);

    BinForRing.push_back(0); //null value for component 0
    BinForRing.push_back(2);
    BinForRing.push_back(5);
    BinForRing.push_back(8);
    BinForRing.push_back(10);
    BinForRing.push_back(12);
    BinForRing.push_back(15);
    BinForRing.push_back(17);
  }
  
  for (size_t i=0;i<SingleExtString.size();i++)
    LogTrace("TkLayerMap") << "[initialize SingleExtString["<<i<<"] " << SingleExtString[i];

  binToDet=(uint32_t*) malloc(nchX*nchY*sizeof(uint32_t));
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
  siStripSubStructure.getTIDDetectors(TkDetIdList,LayerDetIdList,0,layerEnumNb-4,0,0);

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
  siStripSubStructure.getTECDetectors(TkDetIdList,LayerDetIdList,0,layerEnumNb-13,0,0);
  
  LogTrace("TkLayerMap") << "[TkLayerMap::createTEC] layer " << layerEnumNb-13  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xybin=getXY_TEC(LayerDetIdList[j],layerEnumNb);
    binToDet[(xybin.ix-1)+nchX*(xybin.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTEC] " << LayerDetIdList[j]<< " " << xybin.ix << " " << xybin.iy  << " " << xybin.x << " " << xybin.y ;
    
  }
}


const TkLayerMap::XYbin TkLayerMap::getXY(uint32_t& detid, int layerEnumNb){
  LogTrace("TkLayerMap") << "[TkLayerMap::getXY] " << detid << " layer " << layerEnumNb; 

  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  if(layerEnumNb!=layerEnumNb_)
    throw cms::Exception("CorruptedData")
      << "[TkLayerMap::getXY] Fill of DetId " << detid << " layerEnumNb " << layerEnumNb << " are requested to wrong TkLayerMap " << layerEnumNb_ << " \nPlease check the TkDetMap code"; 
 

  if(layerEnumNb<5)  
    return getXY_TIB(detid,layerEnumNb);
  else if(layerEnumNb<8)  
    return getXY_TID(detid,layerEnumNb); 
  else if(layerEnumNb<14)  
    return getXY_TOB(detid,layerEnumNb); 
  else 
    return getXY_TEC(detid,layerEnumNb); 
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
  xybin.iy= int(2. * D.moduleNumber() - (D.isBackRing()?0.:1.));

  xybin.x=lowX+xybin.ix-0.5;
  xybin.y=lowY+xybin.iy-0.5;
  return xybin;
}

TkLayerMap::XYbin TkLayerMap::getXY_TEC(uint32_t& detid, int layerEnumNb){  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TECDetId D(detid);
  xybin.ix=D.isZMinusSide()?18-BinForRing[D.ring()]:BinForRing[D.ring()]+17;
  if(D.isStereo())
    xybin.ix+=(D.isZMinusSide()?-1:1);

  if(D.isZMinusSide()){
    xybin.iy= (D.petalNumber()-1)*(ModulesInRingFront[D.ring()]+ModulesInRingBack[D.ring()]) + ModulesInRingFront[D.ring()] - D.moduleNumber() +1;
    if(D.isBackPetal())
      xybin.iy+=ModulesInRingBack[D.ring()];
  }else{ 
    xybin.iy= (D.petalNumber()-1)*(ModulesInRingFront[D.ring()]+ModulesInRingBack[D.ring()])+D.moduleNumber();
    if(D.isBackPetal())
      xybin.iy+=ModulesInRingFront[D.ring()];
  }

  xybin.x=lowX+xybin.ix-0.5;
  xybin.y=lowY+xybin.iy-0.5;
  return xybin;
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

  TkMap.resize(23);
  //Create TkLayerMap for each layer declared in the TkLayerEnum
  for(int layer=1;layer<23;++layer){
    TkMap[layer]=new TkLayerMap(layer);
  }
}

TkDetMap::~TkDetMap(){
  detmapType::iterator iter=TkMap.begin();
  detmapType::iterator iterE=TkMap.end();
 
  for(;iter!=iterE;++iter)
    delete (*iter);  
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
  }
  cached_XYbin=TkMap[cached_layer]->getXY(detid,layer);
  LogTrace("TkDetMap") <<"[FindLayer] detid "<< detid << " cached_XYbin " << cached_XYbin.ix << " "<< cached_XYbin.iy;

  return cached_layer;
}



void TkDetMap::getComponents(int& layer,
			     int& nchX,double& lowX,double& highX,
			     int& nchY,double& lowY,double& highY){
  nchX=TkMap[layer]->get_nchX();
  lowX=TkMap[layer]->get_lowX();
  highX=TkMap[layer]->get_highX();
  nchY=TkMap[layer]->get_nchY();
  lowY=TkMap[layer]->get_lowY();
  highY=TkMap[layer]->get_highY();
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
