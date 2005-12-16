#ifndef CALIBFORMATS_SISTRIPOBJECTS_SISTRIPCONTROLCONNECTIVITY_H
#define CALIBFORMATS_SISTRIPOBJECTS_SISTRIPCONTROLCONNECTIVITY_H

//#include "DataFormats/DetId/interface/DetId.h"
#include <boost/cstdint.hpp>
#include <vector>
#include <map>

using namespace std;

class SiStripControlConnectivity {
public:

  SiStripControlConnectivity(){}
  ~SiStripControlConnectivity(){}

  uint32_t getDetId(short fec_num,short ring_num,short ccu_num,short i2c_add);
  void getConnectionInfo(uint32_t id,short& fec_num,
               short& ring_num,short& ccu_num,short& i2c_add);

  int getDetIds(short fec_num, short ring_num, 
                short ccu_num, vector<uint32_t>& dets);
  int getDetIds(short fec_num, short ring_num, vector<uint32_t>& dets); 
  int getDetIds(short fec_num, vector<uint32_t>& dets); 


  void setPair(uint32_t det_id, short& fec_num, 
         short& ring_num,short& ccu_num,
         short& i2c_add,vector<short>& apv_adds);
  
  void clean(){theMap.clear();}
  void debug();

 private:

  struct DetControlInfo{
    short fecNumber;
    short ringNumber;
    short ccuAddress;
    short i2cChannel;
    vector<short> apvAddresses;
  };
  typedef map<uint32_t, DetControlInfo> MapType;
  map<uint32_t, DetControlInfo> theMap;  

};

#endif
