#ifndef CalibTracker_SiStripCommon_TKHistoMap_h
#define CalibTracker_SiStripCommon_TKHistoMap_h


#include <map>
#include <boost/cstdint.hpp>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TkLayerMap{

 public:

  struct XYbin{
    XYbin(const XYbin& in){ix=in.ix;iy=in.iy;x=in.x;y=in.y;}
    XYbin(int16_t _ix=-999, int16_t _iy=-999, float _x=-999., float _y=-999.){ix=_ix;iy=_iy;x=_x;y=_y;}
    int16_t ix,iy;
    float x,y;
  };
  
  enum  TkLayerEnum { INVALID=0,
		      TIB_L1,
		      TIB_L2,
		      TIB_L3,
		      TIB_L4,         
		      TID_D1,
		      TID_D2,
		      TID_D3,
		      TOB_L1,
		      TOB_L2,
		      TOB_L3,
		      TOB_L4,
		      TOB_L5,
		      TOB_L6,
		      TEC_W1,
		      TEC_W2,
		      TEC_W3,
		      TEC_W4,
		      TEC_W5,
		      TEC_W6,
		      TEC_W7,
		      TEC_W8,
		      TEC_W9
  };

  
  TkLayerMap(int in);
  ~TkLayerMap(){};
  
  const XYbin& getXY(uint32_t& detid);

  int& get_nchX(){return nchX;}
  int& get_nchY(){return nchY;}
  double& get_lowX(){return lowX;}
  double& get_highX(){return highX;}
  double& get_lowY(){return lowY;}
  double& get_highY(){return highY;}

 private:

  void createTIB12(std::vector<uint32_t>& TkDetIdList, int layer);
  void createTIB34(std::vector<uint32_t>& TkDetIdList, int layer);
  void createTOB12(std::vector<uint32_t>& TkDetIdList, int layer);
  void createTOB36(std::vector<uint32_t>& TkDetIdList, int layer);
  void createTID13 (std::vector<uint32_t>& TkDetIdList,  int layer){createTest(TkDetIdList); }
  //void createTEC(std::vector<uint32_t>& TkDetIdList);
  void createTest(std::vector<uint32_t>& TkDetIdList);

 private:
  std::map<uint32_t,XYbin> DetToBin;
  std::vector<uint32_t> binToDet;

  int nchX;
  int nchY;
  double lowX,highX;
  double lowY, highY;
};

class TkDetMap{

 public:
  TkDetMap();
  TkDetMap(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~TkDetMap();

  const TkLayerMap::XYbin& getXY(uint32_t&);
  std::string getLayerName(int& in);
  int16_t FindLayer(uint32_t& detid);

  void getComponents(int& layer,
		     int& nchX,double& lowX,double& highX,
		     int& nchY,double& lowY,double& highY);
 private:

  void doMe();
  int16_t layerSearch(uint32_t detid);

 private:
  typedef std::map<int,TkLayerMap*> detmapType;
  detmapType TkMap;
  uint32_t cached_detid;
  int16_t cached_layer;
  TkLayerMap::XYbin cached_XYbin;
  detmapType::const_iterator cached_iterator;
};


#endif
