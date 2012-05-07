#ifndef CalibTracker_SiStripCommon_TKHistoMap_h
#define CalibTracker_SiStripCommon_TKHistoMap_h


#include <map>
#include <boost/cstdint.hpp>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

class TkLayerMap{

 public:

  struct XYbin{
    XYbin(const XYbin& in){ix=in.ix;iy=in.iy;x=in.x;y=in.y;}
    XYbin(int16_t _ix=-999, int16_t _iy=-999, float _x=-999., float _y=-999.){ix=_ix;iy=_iy;x=_x;y=_y;}
    int16_t ix,iy;
    float x,y;
  };
  
  enum  TkLayerEnum { INVALID=0,
		      TIB_L1, //1
		      TIB_L2,
		      TIB_L3,
		      TIB_L4,         
		      TIDM_D1, //5
		      TIDM_D2,
		      TIDM_D3,
		      TIDP_D1, //8
		      TIDP_D2,
		      TIDP_D3,
		      TOB_L1, //11
		      TOB_L2,
		      TOB_L3, 
		      TOB_L4,
		      TOB_L5,
		      TOB_L6,
		      TECM_W1, //17
		      TECM_W2,
		      TECM_W3,
		      TECM_W4,
		      TECM_W5,
		      TECM_W6,
		      TECM_W7,
		      TECM_W8,
		      TECM_W9,
		      TECP_W1, //26
		      TECP_W2,
		      TECP_W3,
		      TECP_W4,
		      TECP_W5,
		      TECP_W6,
		      TECP_W7,
		      TECP_W8,
		      TECP_W9 //34
  };

  
  TkLayerMap(int in);
  ~TkLayerMap(){
    delete [] binToDet;
  };
  
  const XYbin getXY(uint32_t& detid, int layerEnumNb=0);

  int& get_nchX(){return nchX;}
  int& get_nchY(){return nchY;}
  double& get_lowX(){return lowX;}
  double& get_highX(){return highX;}
  double& get_lowY(){return lowY;}
  double& get_highY(){return highY;}

  static const int16_t layerSearch(uint32_t detid);

  uint32_t getDetFromBin(int ix, int iy);
  uint32_t* getBinToDet(){return binToDet;}

 private:

  XYbin getXY_TIB(uint32_t& detid, int layerEnumNb=0);
  XYbin getXY_TOB(uint32_t& detid, int layerEnumNb=0);
  XYbin getXY_TID(uint32_t& detid, int layerEnumNb=0);
  XYbin getXY_TEC(uint32_t& detid, int layerEnumNb=0);

  void initialize(int layer);

  void createTIB(std::vector<uint32_t>& TkDetIdList, int layer);
  void createTOB(std::vector<uint32_t>& TkDetIdList, int layer);
  void createTID(std::vector<uint32_t>& TkDetIdList, int layer);
  void createTEC(std::vector<uint32_t>& TkDetIdList, int layer);

 private:
  uint32_t* binToDet;
  XYbin xybin;

  int layerEnumNb_; //In the enumerator sequence
  int nchX;
  int nchY;
  double lowX,highX;
  double lowY, highY;

  std::vector<uint32_t> SingleExtString,ModulesInRingFront,ModulesInRingBack,BinForRing;
  uint32_t Nstring_ext, Nrod, Offset;

};

class TkDetMap{

 public:
  TkDetMap();
  TkDetMap(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~TkDetMap();

  const TkLayerMap::XYbin& getXY(uint32_t&);
  std::string getLayerName(int& in);
  int getLayerNum(std::string& in);
  void getSubDetLayerSide(int& in,SiStripDetId::SubDetector&,uint32_t& layer,uint32_t& side);

  int16_t FindLayer(uint32_t& detid);

  void getComponents(int& layer,
		     int& nchX,double& lowX,double& highX,
		     int& nchY,double& lowY,double& highY);
 
  uint32_t getDetFromBin(int layer, int ix, int iy){ return TkMap[layer]->getDetFromBin(ix,iy); }
  uint32_t getDetFromBin(std::string layerName, int ix, int iy){return getDetFromBin(getLayerNum(layerName),ix,iy);}

  void getDetsForLayer(int layer,std::vector<uint32_t>& output);

 private:

  void doMe();

 private:
  typedef std::vector<TkLayerMap*> detmapType;
  detmapType TkMap;
  uint32_t cached_detid;
  int16_t cached_layer;
  TkLayerMap::XYbin cached_XYbin;
};


#endif
