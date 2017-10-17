#ifndef GEMRawToDigi_GEMROMAO_H
#define GEMRawToDigi_GEMROMAO_H
#include <map>
class GEMROmap{
 public:
  struct eCoord{
    int chamberId;
    int vfatId;
    int channelId;
    bool operator < (const eCoord& r) const{
      if (chamberId == r.chamberId){
	if ( vfatId == r.vfatId){
	  return channelId < r.channelId;
	}else{
	  return vfatId<r.vfatId;
	}
      }else{
	return chamberId < r.chamberId;
      }
    }
  };
  struct dCoord{
    int etaId;
    int stripId;
  };

 public:
  GEMROmap(){};
  const dCoord& hitPosition(const eCoord& ec){return roMap[ec];}
  
  void add(eCoord e,dCoord d) {roMap[e]=d;}
 private:
  std::map<eCoord,dCoord> roMap;
  
};
#endif
