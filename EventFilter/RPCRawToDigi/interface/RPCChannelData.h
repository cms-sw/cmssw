#ifndef RPCChannelData_h
#define RPCChannelData_h


/** \class RPCChannelData 
 * Unpacks RPC Channel Data Record (needs pointer to beginning of buffer)
 *
 *  $Date: 2005/11/24 18:06:23 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
 

#include <vector>

using namespace std;

class RPCChannelData {

public:
  
  /// Constructor
  RPCChannelData(const unsigned int* index);

  /// Destructor
  virtual ~RPCChannelData() {}

  /// unpacked data access methods
  int channel();
  int tbRmb();

  //vector<ChamberData> chambData(){return chambersData;}


  static const int CHANNEL_MASK  = 0X1F;
  static const int CHANNEL_SHIFT =0;

  static const int TB_RMB_MASK = 0X3F;
  static const int TB_RMB_SHIFT =5;

private:

  const unsigned int * word_;
 
  int channel_;
  int tbRmb_;
  
  //vector<ChamberData> chambersData;

};




#endif
