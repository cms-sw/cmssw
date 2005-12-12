#ifndef RPCBXData_h
#define RPCBXData_h


/** \class RPCBXData
 *
 * Class for unpacking "Start of BX Data" record in RPC raw data
 * (needs pointer to buffer).
 *  
 *  $Date: 2005/11/11 16:23:56 $
 *  $Revision: 1.1 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */
 

#include <vector>

using namespace std;

class RPCBXData {

public:
  
  /// Constructor
  RPCBXData(const unsigned int* index);

  /// Destructor
  virtual ~RPCBXData() {}

  /// unpacked data access methods
  int bx();

  //vector<RPCChannelData> chanData(){return channelsData;}
  
  static const int BX_MASK  = 0XC;
  static const int BX_SHIFT = 0;

private:

  const unsigned int * word_;
 
  int bx_;
  
  //vector<RPCChannelData> channelsData;

};




#endif
