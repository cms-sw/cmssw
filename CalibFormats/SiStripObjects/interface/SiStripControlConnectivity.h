#ifndef CALIBFORMATS_SISTRIPOBJECTS_SISTRIPCONTROLCONNECTIVITY_H
#define CALIBFORMATS_SISTRIPOBJECTS_SISTRIPCONTROLCONNECTIVITY_H

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <map>
using namespace std;

class SiStripControlConnectivity {
public:

  SiStripControlConnectivity(){}
  ~SiStripControlConnectivity(){}

  DetId getDetId(short fec_num,short ring_num,short ccu_num,short i2c_add);
  void getConnectionInfo(DetId id,short& fec_num,
               short& ring_num,short& ccu_num,short& i2c_add);

  int getDetIds(short fec_num, short ring_num, 
                short ccu_num, vector<DetId>& dets);
  int getDetIds(short fec_num, short ring_num, vector<DetId>& dets); 
  int getDetIds(short fec_num, vector<DetId>& dets); 


  void setPair(DetId det_id, short& fec_num, 
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
  typedef map<DetId, DetControlInfo> MapType;
  map<DetId, DetControlInfo> theMap;  

};

#endif
