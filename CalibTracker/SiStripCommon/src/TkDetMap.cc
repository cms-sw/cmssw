#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cstring>

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
  
  switch (layerEnumNb_)
    {
    case TkLayerMap::TIB_L1:
    case TkLayerMap::TIB_L2:
    case TkLayerMap::TIB_L3:
    case TkLayerMap::TIB_L4:         
      createTIB(TkDetIdList,layerEnumNb_);
      break;
    case TkLayerMap::TIDP_D1:
    case TkLayerMap::TIDP_D2:
    case TkLayerMap::TIDP_D3:
    case TkLayerMap::TIDM_D1:
    case TkLayerMap::TIDM_D2:
    case TkLayerMap::TIDM_D3:
      createTID(TkDetIdList,layerEnumNb_); 
      break;
    case TkLayerMap::TOB_L1:
    case TkLayerMap::TOB_L2:
    case TkLayerMap::TOB_L3:
    case TkLayerMap::TOB_L4:
    case TkLayerMap::TOB_L5:
    case TkLayerMap::TOB_L6:
      createTOB(TkDetIdList,layerEnumNb_); 
    break;
    case TkLayerMap::TECP_W1:
    case TkLayerMap::TECP_W2:
    case TkLayerMap::TECP_W3:
    case TkLayerMap::TECP_W4:
    case TkLayerMap::TECP_W5: 
    case TkLayerMap::TECP_W6:
    case TkLayerMap::TECP_W7:
    case TkLayerMap::TECP_W8:
    case TkLayerMap::TECP_W9:
    case TkLayerMap::TECM_W1:
    case TkLayerMap::TECM_W2:
    case TkLayerMap::TECM_W3:
    case TkLayerMap::TECM_W4:
    case TkLayerMap::TECM_W5: 
    case TkLayerMap::TECM_W6:
    case TkLayerMap::TECM_W7:
    case TkLayerMap::TECM_W8:
    case TkLayerMap::TECM_W9:
      createTEC(TkDetIdList,layerEnumNb_); 
      break;
    default:
      edm::LogError("TkLayerMap") <<" TkLayerMap::requested creation of a wrong layer Nb "<< layerEnumNb_; 
    }
}

uint32_t TkLayerMap::getDetFromBin(int ix, int iy) const {
  
  int val=(ix-1)+nchX*(iy-1);
  if(val>-1 && val < nchX*nchY)
    return binToDet[val];
  return 0;
}

const int16_t TkLayerMap::layerSearch(uint32_t detid) {
  
    //  switch((detid>>25)&0x7){
  if(SiStripDetId(detid).subDetector()==SiStripDetId::TIB){
    TIBDetId D(detid);
    return TkLayerMap::TIB_L1  -1 +D.layerNumber();
  } else if (SiStripDetId(detid).subDetector()==SiStripDetId::TID){
    TIDDetId D(detid);
    return TkLayerMap::TIDM_D1 -1 + (D.side() -1)*3 + D.wheel();
  } else if (SiStripDetId(detid).subDetector()==SiStripDetId::TOB){
    TOBDetId D(detid);
    return TkLayerMap::TOB_L1  -1 +D.layerNumber();
  } else if (SiStripDetId(detid).subDetector()==SiStripDetId::TEC){
    TECDetId D(detid);
    return TkLayerMap::TECM_W1 -1 + (D.side() -1)*9 + D.wheel();
  }
  return 0;
}


void TkLayerMap::initialize(int layer){

  switch (layer){
  case TkLayerMap::TIB_L1: //TIBL1
    
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
  case TkLayerMap::TIB_L2:
    
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
  case TkLayerMap::TIB_L3:

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
  case TkLayerMap::TIB_L4:

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
  case TkLayerMap::TIDM_D1:  //TID
  case TkLayerMap::TIDM_D2:  //TID
  case TkLayerMap::TIDM_D3:  //TID

    nchX=7;
    lowX=-7.;
    highX=0.;
    nchY=40;
    lowY=0.0;
    highY=1.*nchY;

    break;
  case TkLayerMap::TIDP_D1:  //TID
  case TkLayerMap::TIDP_D2:  //TID
  case TkLayerMap::TIDP_D3:  //TID
    
    nchX=7;
    lowX=0.;
    highX=7.;
    nchY=40;
    lowY=0.0;
    highY=1.*nchY;

    break;
  case TkLayerMap::TOB_L1:  //TOBL1

    Nrod=42;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=2*(Nrod+1);
    lowY=-1.*(Nrod+1.);
    highY=(Nrod+1.);

    break;
  case TkLayerMap::TOB_L2:
    
    Nrod=48;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=int(2.*(Nrod+1.));
    lowY=-1.*(Nrod+1.);
    highY=(Nrod+1.);
    
    break;
  case TkLayerMap::TOB_L3: //TOBL3
   
    Nrod=54;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;
    
    break;
  case TkLayerMap::TOB_L4:
    
    Nrod=60;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;

    break;
  case TkLayerMap::TOB_L5:
    
    Nrod=66;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;

    break;
  case TkLayerMap::TOB_L6:
    
    Nrod=74;
    nchX=12;
    lowX=-6.;
    highX=6.;
    nchY=Nrod;
    lowY=0.;
    highY=1.*Nrod;

    break;
  default: //TEC
    switch (layer){
    case TkLayerMap::TECM_W1:
    case TkLayerMap::TECM_W2:
    case TkLayerMap::TECM_W3:
      nchX=16;
      lowX=-16.;
      highX=0.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(1);
      BinForRing.push_back(4);
      BinForRing.push_back(7);
      BinForRing.push_back(9);
      BinForRing.push_back(11);
      BinForRing.push_back(14);
      BinForRing.push_back(16);
      break;
    case TkLayerMap::TECM_W4:
    case TkLayerMap::TECM_W5: 
    case TkLayerMap::TECM_W6:
      nchX=13;
      lowX=-16.;
      highX=-3.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(0);
      BinForRing.push_back(1);
      BinForRing.push_back(4);
      BinForRing.push_back(6);
      BinForRing.push_back(8);
      BinForRing.push_back(11);
      BinForRing.push_back(13);
      break;
    case TkLayerMap::TECM_W7:
    case TkLayerMap::TECM_W8:
      nchX=10;
      lowX=-16.;
      highX=-6.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(0);
      BinForRing.push_back(0);
      BinForRing.push_back(1);
      BinForRing.push_back(3);
      BinForRing.push_back(5);
      BinForRing.push_back(8);
      BinForRing.push_back(10);
      break;
    case TkLayerMap::TECM_W9:
      nchX=8;
      lowX=-16.;
      highX=-8.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(0);
      BinForRing.push_back(0);
      BinForRing.push_back(0);
      BinForRing.push_back(1);
      BinForRing.push_back(3);
      BinForRing.push_back(6);
      BinForRing.push_back(8);
      break;
    case TkLayerMap::TECP_W1:
    case TkLayerMap::TECP_W2:
    case TkLayerMap::TECP_W3:
      nchX=16;
      lowX=0.;
      highX=16.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(1);
      BinForRing.push_back(4);
      BinForRing.push_back(7);
      BinForRing.push_back(9);
      BinForRing.push_back(11);
      BinForRing.push_back(14);
      BinForRing.push_back(16);
      break;
    case TkLayerMap::TECP_W4:
    case TkLayerMap::TECP_W5: 
    case TkLayerMap::TECP_W6:
      nchX=13;
      lowX=3.;
      highX=16.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(0);
      BinForRing.push_back(1);
      BinForRing.push_back(4);
      BinForRing.push_back(6);
      BinForRing.push_back(8);
      BinForRing.push_back(11);
      BinForRing.push_back(13);
      break;
    case TkLayerMap::TECP_W7:
    case TkLayerMap::TECP_W8:
      nchX=10;
      lowX=6.;
      highX=16.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(0);
      BinForRing.push_back(0);
      BinForRing.push_back(1);
      BinForRing.push_back(3);
      BinForRing.push_back(5);
      BinForRing.push_back(8);
      BinForRing.push_back(10);
      break;
    case TkLayerMap::TECP_W9:
      nchX=8;
      lowX=8.;
      highX=16.;
      nchY=80;
      lowY=0.;
      highY=1.*nchY;

      BinForRing.push_back(0); //null value for component 0
      BinForRing.push_back(0);
      BinForRing.push_back(0);
      BinForRing.push_back(0);
      BinForRing.push_back(1);
      BinForRing.push_back(3);
      BinForRing.push_back(6);
      BinForRing.push_back(8);
    }


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
  }
  
  for (size_t i=0;i<SingleExtString.size();i++)
    LogTrace("TkLayerMap") << "[initialize SingleExtString["<<i<<"] " << SingleExtString[i];

  binToDet= new uint32_t[nchX*nchY];
  for(size_t i=0;i<(size_t) nchX*nchY;++i)
    binToDet[i]=0;
}
 
void TkLayerMap::createTIB(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTIBDetectors(TkDetIdList,LayerDetIdList,layerEnumNb,0,0,0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTIB12] layer " << layerEnumNb  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  XYbin xyb;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xyb=getXY_TIB(LayerDetIdList[j],layerEnumNb);
    binToDet[(xyb.ix-1)+nchX*(xyb.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTIB] " << LayerDetIdList[j]<< " " << xyb.ix << " " << xyb.iy  << " " << xyb.x << " " << xyb.y ;
  }
}

void TkLayerMap::createTOB(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTOBDetectors(TkDetIdList,LayerDetIdList,layerEnumNb-10,0,0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTOB] layer " << layerEnumNb-10  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  XYbin xyb;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xyb=getXY_TOB(LayerDetIdList[j],layerEnumNb);
    binToDet[(xyb.ix-1)+nchX*(xyb.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTOB] " << LayerDetIdList[j]<< " " << xyb.ix << " " << xyb.iy  << " " << xyb.x << " " << xyb.y ;
  }
}

void TkLayerMap::createTID(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTIDDetectors(TkDetIdList,LayerDetIdList,(layerEnumNb-TkLayerMap::TIDM_D1)/3+1,(layerEnumNb-TkLayerMap::TIDM_D1)%3+1,0,0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTID] layer side " << (layerEnumNb-TkLayerMap::TIDM_D1)/3+1 << " nb " << (layerEnumNb-TkLayerMap::TIDM_D1)%3+1  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;
  
  XYbin xyb;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xyb=getXY_TID(LayerDetIdList[j],layerEnumNb);
    binToDet[(xyb.ix-1)+nchX*(xyb.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTID] " << LayerDetIdList[j]<< " " << xyb.ix << " " << xyb.iy  << " " << xyb.x << " " << xyb.y ;
  }
}

void TkLayerMap::createTEC(std::vector<uint32_t>& TkDetIdList,int layerEnumNb){
  
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure siStripSubStructure;
  
  //extract  vector of module in the layer
  siStripSubStructure.getTECDetectors(TkDetIdList,LayerDetIdList,(layerEnumNb-TkLayerMap::TECM_W1)/9+1,(layerEnumNb-TkLayerMap::TECM_W1)%9+1,0,0);
  
  LogTrace("TkLayerMap") << "[TkLayerMap::createTEC] layer side " << (layerEnumNb-TkLayerMap::TECM_W1)/9+1 << " " << (layerEnumNb-TkLayerMap::TECM_W1)%9+1  << " number of dets " << LayerDetIdList.size() << " lowY " << lowY << " high " << highY << " Nstring " << Nstring_ext;

  XYbin xyb;

  for(size_t j=0;j<LayerDetIdList.size();++j){
    xyb=getXY_TEC(LayerDetIdList[j],layerEnumNb);
    binToDet[(xyb.ix-1)+nchX*(xyb.iy-1)]=LayerDetIdList[j];
    LogTrace("TkLayerMap") << "[TkLayerMap::createTEC] " << LayerDetIdList[j]<< " " << xyb.ix << " " << xyb.iy  << " " << xyb.x << " " << xyb.y ;
    
  }
}


const TkLayerMap::XYbin TkLayerMap::getXY(uint32_t detid, int layerEnumNb) const {
  LogTrace("TkLayerMap") << "[TkLayerMap::getXY] " << detid << " layer " << layerEnumNb; 

  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  if(layerEnumNb!=layerEnumNb_)
    throw cms::Exception("CorruptedData")
      << "[TkLayerMap::getXY] Fill of DetId " << detid << " layerEnumNb " << layerEnumNb << " are requested to wrong TkLayerMap " << layerEnumNb_ << " \nPlease check the TkDetMap code"; 
 

  if(layerEnumNb>=TkLayerMap::TIB_L1 && layerEnumNb<=TkLayerMap::TIB_L4)  
    return getXY_TIB(detid,layerEnumNb);
  else if(layerEnumNb>=TkLayerMap::TIDM_D1 && layerEnumNb<=TkLayerMap::TIDP_D3)  
    return getXY_TID(detid,layerEnumNb); 
  else if(layerEnumNb>=TkLayerMap::TOB_L1 && layerEnumNb<=TkLayerMap::TOB_L6)  
    return getXY_TOB(detid,layerEnumNb); 
  else 
    return getXY_TEC(detid,layerEnumNb); 
}

uint32_t TkLayerMap::get_Offset( TIBDetId D ) const {
  if(D.layerNumber()%2)
    return D.isInternalString()?2:1;
  else
    return D.isInternalString()?1:2;
}


TkLayerMap::XYbin TkLayerMap::getXY_TIB(uint32_t detid, int layerEnumNb) const {  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TIBDetId D(detid);

  XYbin xyb;
  xyb.ix=2*(D.isZMinusSide()?-1*D.moduleNumber()+3:D.moduleNumber()+2)+get_Offset(D);
  xyb.iy=D.isInternalString()?D.stringNumber()+SingleExtString[D.stringNumber()]:D.stringNumber();
  if(D.layerNumber()<3 && !D.isStereo())
    xyb.iy+=Nstring_ext+2;
  
  xyb.x=lowX+xyb.ix-0.5;
  xyb.y=lowY+xyb.iy-0.5;
  return xyb;

}

TkLayerMap::XYbin TkLayerMap::getXY_TOB(uint32_t detid, int layerEnumNb) const {  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TOBDetId D(detid);

  XYbin xyb;
  xyb.ix=D.isZMinusSide()?-1*D.moduleNumber()+7:D.moduleNumber()+6;
  xyb.iy=D.rodNumber();  
  if(D.layerNumber()<3 && !D.isStereo())
    xyb.iy+=Nrod+2;

  xyb.x=lowX+xyb.ix-0.5;
  xyb.y=lowY+xyb.iy-0.5;
  return xyb;
}

TkLayerMap::XYbin TkLayerMap::getXY_TID(uint32_t detid, int layerEnumNb) const {  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TIDDetId D(detid);

  XYbin xyb;

  xyb.ix=D.isZMinusSide()?-3*D.ring()+10:3*D.ring()-2;
  if(D.isStereo())
    xyb.ix+=(D.isZMinusSide()?-1:1);
  xyb.iy= int(2. * D.moduleNumber() - (D.isBackRing()?0.:1.));

  xyb.x=lowX+xyb.ix-0.5;
  xyb.y=lowY+xyb.iy-0.5;
  return xyb;
}

TkLayerMap::XYbin TkLayerMap::getXY_TEC(uint32_t detid, int layerEnumNb) const {  
  if(!layerEnumNb)
    layerEnumNb=layerSearch(detid);

  TECDetId D(detid);

  XYbin xyb;

  xyb.ix=D.isZMinusSide()?BinForRing[7]-BinForRing[D.ring()]+1:BinForRing[D.ring()]; //after the introduction of plus and minus histos, the BinForRing should have been changed. on the contrary we hack this part of the code 
  if(D.isStereo())
    xyb.ix+=(D.isZMinusSide()?-1:1);

  if(D.isZMinusSide()){
    xyb.iy= (D.petalNumber()-1)*(ModulesInRingFront[D.ring()]+ModulesInRingBack[D.ring()]) + ModulesInRingFront[D.ring()] - D.moduleNumber() +1;
    if(D.isBackPetal())
      xyb.iy+=ModulesInRingBack[D.ring()];
  }else{ 
    xyb.iy= (D.petalNumber()-1)*(ModulesInRingFront[D.ring()]+ModulesInRingBack[D.ring()])+D.moduleNumber();
    if(D.isBackPetal())
      xyb.iy+=ModulesInRingFront[D.ring()];
  }

  xyb.x=lowX+xyb.ix-0.5;
  xyb.y=lowY+xyb.iy-0.5;
  return xyb;
}

//--------------------------------------

TkDetMap::TkDetMap(const edm::ParameterSet& p,const edm::ActivityRegistry& a){
  doMe();
}

TkDetMap::TkDetMap(){
  doMe();
}

void TkDetMap::doMe() {
  LogTrace("TkDetMap") <<"TkDetMap::constructor ";

  TkMap.resize(35);
  //Create TkLayerMap for each layer declared in the TkLayerEnum
  for(int layer=1;layer<35;++layer){
    TkMap[layer]=new TkLayerMap(layer);
  }
}

TkDetMap::~TkDetMap(){
  detmapType::iterator iter=TkMap.begin();
  detmapType::iterator iterE=TkMap.end();
 
  for(;iter!=iterE;++iter)
    delete (*iter);  
}

const TkLayerMap::XYbin& TkDetMap::getXY(uint32_t& detid , uint32_t& cached_detid , int16_t& cached_layer , TkLayerMap::XYbin& cached_XYbin) const {
  LogTrace("TkDetMap") <<"[getXY] detid "<< detid << " cache " << cached_detid << " layer " << cached_layer << " XY " << cached_XYbin.ix << " " << cached_XYbin.iy  << " " << cached_XYbin.x << " " << cached_XYbin.y ;    
  if(detid==cached_detid)
    return cached_XYbin;

  /*FIXME*/
  //if (layer!=INVALID)
  FindLayer(detid , cached_detid , cached_layer , cached_XYbin );
  LogTrace("TkDetMap") <<"[getXY] detid "<< detid << " cache " << cached_detid << " layer " << cached_layer << " XY " << cached_XYbin.ix << " " << cached_XYbin.iy  << " " << cached_XYbin.x << " " << cached_XYbin.y ;    
  return cached_XYbin;
}

int16_t TkDetMap::FindLayer(uint32_t& detid , uint32_t& cached_detid , int16_t& cached_layer , TkLayerMap::XYbin& cached_XYbin) const { 

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



void TkDetMap::getComponents(int layer,
			     int& nchX,double& lowX,double& highX,
			     int& nchY,double& lowY,double& highY) const {
  nchX=TkMap[layer]->get_nchX();
  lowX=TkMap[layer]->get_lowX();
  highX=TkMap[layer]->get_highX();
  nchY=TkMap[layer]->get_nchY();
  lowY=TkMap[layer]->get_lowY();
  highY=TkMap[layer]->get_highY();
}

void TkDetMap::getDetsForLayer(int layer,std::vector<uint32_t>& output) const {
  output.clear();
  size_t size_=TkMap[layer]->get_nchX()*TkMap[layer]->get_nchY();
  output.resize(size_);
  memcpy((void*)&output[0],(void*)TkMap[layer]->getBinToDet(),size_*sizeof(uint32_t));
}

std::string TkDetMap::getLayerName(int& in) const {
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
    case TkLayerMap::TIDP_D1:
      return "TIDP_D1";
    case TkLayerMap::TIDP_D2:
      return "TIDP_D2";
    case TkLayerMap::TIDP_D3:
      return "TIDP_D3";
    case TkLayerMap::TIDM_D1:
      return "TIDM_D1";
    case TkLayerMap::TIDM_D2:
      return "TIDM_D2";
    case TkLayerMap::TIDM_D3:
      return "TIDM_D3";
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
    case TkLayerMap::TECP_W1:
      return "TECP_W1";
    case TkLayerMap::TECP_W2:
      return "TECP_W2";
    case TkLayerMap::TECP_W3:
      return "TECP_W3";
    case TkLayerMap::TECP_W4:
      return "TECP_W4";
    case TkLayerMap::TECP_W5: 
      return "TECP_W5";
    case TkLayerMap::TECP_W6:
      return "TECP_W6";
    case TkLayerMap::TECP_W7:
      return "TECP_W7";
    case TkLayerMap::TECP_W8:
      return "TECP_W8";
    case TkLayerMap::TECP_W9:
      return "TECP_W9";
    case TkLayerMap::TECM_W1:
      return "TECM_W1";
    case TkLayerMap::TECM_W2:
      return "TECM_W2";
    case TkLayerMap::TECM_W3:
      return "TECM_W3";
    case TkLayerMap::TECM_W4:
      return "TECM_W4";
    case TkLayerMap::TECM_W5: 
      return "TECM_W5";
    case TkLayerMap::TECM_W6:
      return "TECM_W6";
    case TkLayerMap::TECM_W7:
      return "TECM_W7";
    case TkLayerMap::TECM_W8:
      return "TECM_W8";
    case TkLayerMap::TECM_W9:
      return "TECM_W9";
    }
  return "Invalid";
}

int TkDetMap::getLayerNum(const std::string& in) const {
  if(in.compare( "TIB_L1")==0)
    return TkLayerMap::TIB_L1;
  if(in.compare( "TIB_L2")==0)
    return TkLayerMap::TIB_L2;
  if(in.compare( "TIB_L3")==0)
    return TkLayerMap::TIB_L3;
  if(in.compare( "TIB_L4")==0)         
    return TkLayerMap::TIB_L4;         
  if(in.compare( "TIDP_D1")==0)
    return TkLayerMap::TIDP_D1;
  if(in.compare( "TIDP_D2")==0)
    return TkLayerMap::TIDP_D2;
  if(in.compare( "TIDP_D3")==0)
    return TkLayerMap::TIDP_D3;
  if(in.compare( "TIDM_D1")==0)
    return TkLayerMap::TIDM_D1;
  if(in.compare( "TIDM_D2")==0)
    return TkLayerMap::TIDM_D2;
  if(in.compare( "TIDM_D3")==0)
    return TkLayerMap::TIDM_D3;
  if(in.compare( "TOB_L1")==0)
    return TkLayerMap::TOB_L1;
  if(in.compare( "TOB_L2")==0)
    return TkLayerMap::TOB_L2;
  if(in.compare( "TOB_L3")==0)
    return TkLayerMap::TOB_L3;
  if(in.compare( "TOB_L4")==0)
    return TkLayerMap::TOB_L4;
  if(in.compare( "TOB_L5")==0)
    return TkLayerMap::TOB_L5;
  if(in.compare( "TOB_L6")==0)
    return TkLayerMap::TOB_L6;
  if(in.compare( "TECP_W1")==0)
    return TkLayerMap::TECP_W1;
  if(in.compare( "TECP_W2")==0)
    return TkLayerMap::TECP_W2;
  if(in.compare( "TECP_W3")==0)
    return TkLayerMap::TECP_W3;
  if(in.compare( "TECP_W4")==0)
    return TkLayerMap::TECP_W4;
  if(in.compare( "TECP_W5")==0)
    return TkLayerMap::TECP_W5; 
  if(in.compare( "TECP_W6")==0)
    return TkLayerMap::TECP_W6;
  if(in.compare( "TECP_W7")==0)
    return TkLayerMap::TECP_W7;
  if(in.compare( "TECP_W8")==0)
    return TkLayerMap::TECP_W8;
  if(in.compare( "TECP_W9")==0)
    return TkLayerMap::TECP_W9;
  if(in.compare( "TECM_W1")==0)
    return TkLayerMap::TECM_W1;
  if(in.compare( "TECM_W2")==0)
    return TkLayerMap::TECM_W2;
  if(in.compare( "TECM_W3")==0)
    return TkLayerMap::TECM_W3;
  if(in.compare( "TECM_W4")==0)
    return TkLayerMap::TECM_W4;
  if(in.compare( "TECM_W5")==0)
    return TkLayerMap::TECM_W5; 
  if(in.compare( "TECM_W6")==0)
    return TkLayerMap::TECM_W6;
  if(in.compare( "TECM_W7")==0)
    return TkLayerMap::TECM_W7;
  if(in.compare( "TECM_W8")==0)
    return TkLayerMap::TECM_W8;
  if(in.compare( "TECM_W9")==0)
    return TkLayerMap::TECM_W9;
  return 0;
}

void TkDetMap::getSubDetLayerSide(int& in,SiStripDetId::SubDetector& subDet,uint32_t& layer,uint32_t& side) const {
  switch (in)
    {
    case TkLayerMap::TIB_L1:
      subDet = SiStripDetId::TIB;
      layer = 1;
      break;
    case TkLayerMap::TIB_L2:
      subDet = SiStripDetId::TIB;
      layer = 2;
      break;
    case TkLayerMap::TIB_L3:
      subDet = SiStripDetId::TIB;
      layer = 3;
      break;
    case TkLayerMap::TIB_L4:
      subDet = SiStripDetId::TIB;
      layer = 4;
      break;
    case TkLayerMap::TIDP_D1:
      subDet = SiStripDetId::TID;
      layer = 1;
      side=2;
      break;
    case TkLayerMap::TIDP_D2:
      subDet = SiStripDetId::TID;
      layer = 2;
      side=2;
      break;
    case TkLayerMap::TIDP_D3:
      subDet = SiStripDetId::TID;
      layer = 3;
      side=2;
      break;
    case TkLayerMap::TIDM_D1:
      subDet = SiStripDetId::TID;
      layer = 1;
      side=1;
      break;
    case TkLayerMap::TIDM_D2:
      subDet = SiStripDetId::TID;
      layer = 2;
      side=1;
      break;
    case TkLayerMap::TIDM_D3:
      subDet = SiStripDetId::TID;
      layer = 3;
      side=1;
      break;
    case TkLayerMap::TOB_L1:
      subDet = SiStripDetId::TOB;
      layer = 1;
      break;
    case TkLayerMap::TOB_L2:
      subDet = SiStripDetId::TOB;
      layer = 2;
      break;
    case TkLayerMap::TOB_L3:
      subDet = SiStripDetId::TOB;
      layer = 3;
      break;
    case TkLayerMap::TOB_L4:
      subDet = SiStripDetId::TOB;
      layer = 4;
      break;
    case TkLayerMap::TOB_L5:
      subDet = SiStripDetId::TOB;
      layer = 5;
      break;
    case TkLayerMap::TOB_L6:
      subDet = SiStripDetId::TOB;
      layer = 6;
      break;
    case TkLayerMap::TECP_W1:
      subDet = SiStripDetId::TEC;
      layer = 1;
      side=2;
      break;
    case TkLayerMap::TECP_W2:
      subDet = SiStripDetId::TEC;
      layer = 2;
      side=2;
      break;
    case TkLayerMap::TECP_W3:
      subDet = SiStripDetId::TEC;
      layer = 3;
      side=2;
      break;
    case TkLayerMap::TECP_W4:
      subDet = SiStripDetId::TEC;
      layer = 4;
      side=2;
      break;
    case TkLayerMap::TECP_W5: 
      subDet = SiStripDetId::TEC;
      layer = 5;
      side=2;
      break;
    case TkLayerMap::TECP_W6:
      subDet = SiStripDetId::TEC;
      layer = 6;
      side=2;
      break;
    case TkLayerMap::TECP_W7:
      subDet = SiStripDetId::TEC;
      layer = 7;
      side=2;
      break;
    case TkLayerMap::TECP_W8:
      subDet = SiStripDetId::TEC;
      layer = 8;
      side=2;
      break;
    case TkLayerMap::TECP_W9:
      subDet = SiStripDetId::TEC;
      layer = 9;
      side=2;
      break;
    case TkLayerMap::TECM_W1:
      subDet = SiStripDetId::TEC;
      layer = 1;
      side=1;
      break;
    case TkLayerMap::TECM_W2:
      subDet = SiStripDetId::TEC;
      layer = 2;
      side=1;
      break;
    case TkLayerMap::TECM_W3:
      subDet = SiStripDetId::TEC;
      layer = 3;
      side=1;
      break;
    case TkLayerMap::TECM_W4:
      subDet = SiStripDetId::TEC;
      layer = 4;
      side=1;
      break;
    case TkLayerMap::TECM_W5: 
      subDet = SiStripDetId::TEC;
      layer = 5;
      side=1;
      break;
    case TkLayerMap::TECM_W6:
      subDet = SiStripDetId::TEC;
      layer = 6;
      side=1;
      break;
    case TkLayerMap::TECM_W7:
      subDet = SiStripDetId::TEC;
      layer = 7;
      side=1;
      break;
    case TkLayerMap::TECM_W8:
      subDet = SiStripDetId::TEC;
      layer = 8;
      side=1;
      break;
    case TkLayerMap::TECM_W9:
      subDet = SiStripDetId::TEC;
      layer = 9;
      side=1;
      break;
    }
}

//  LocalWords:  TkLayerMap
