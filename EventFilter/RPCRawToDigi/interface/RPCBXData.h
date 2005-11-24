#ifndef RPCBXData_h
#define RPCBXData_h


/** \file
 * BX number
 *
 *  $Date: 2005/11/09 11:37:16 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
 

#include <vector>
#include <EventFilter/RPCRawToDigi/interface/RPCChannelData.h>

using namespace std;

class RPCBXData {

public:
  
  /// Constructor
  RPCBXData(const unsigned char* index);

  /// Destructor
  virtual ~RPCBXData() {}

  /// unpacked data access methods
  int bx();

  vector<RPCChannelData> chanData(){return channelsData;}
  
  static const int BX_MASK  = 0XC;
  static const int BX_SHIFT = 0;

private:

  const unsigned int * word_;
 
  int bx_;
  
  vector<RPCChannelData> channelsData;

};




#endif
