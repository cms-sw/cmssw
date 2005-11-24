#ifndef RPCChannelData_h
#define RPCChannelData_h


/** \file
 * Unpacks Firs record of Channel Data
 *
 *  $Date: 2005/11/07 15:43:50 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */
 
#include <EventFilter/RPCRawToDigi/interface/ChamberData.h>

#include <vector>

using namespace std;

class RPCChannelData {

public:
  
  /// Constructor
  RPCChannelData(const unsigned char* index);

  /// Destructor
  virtual ~RPCChannelData() {}

  /// unpacked data access methods
  int channel();
  int tbRmb();
  int chamber();

  vector<ChamberData> chambData(){return chambersData;}


  static const int CHANNEL_MASK  = 0X1F;
  static const int CHANNEL_SHIFT =0;

  static const int TB_RMB_MASK = 0X3F;
  static const int TB_RMB_SHIFT =5;

  static const int CHAMBER_MASK = 0X3;
  static const int CHAMBER_SHIFT =14;



private:

  const unsigned int * word_;
 
  int channel_;
  int tbRmb_;
  int chamber_;
  
  vector<ChamberData> chambersData;
};




#endif
