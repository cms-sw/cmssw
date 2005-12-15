#ifndef RPCEventData_h
#define RPCEventData_h

/** \class RPCEventData
 * Container for Unpacked RPC Data, used by DQM module
 *
 *  $Date: 2005/12/12 17:29:04 $
 *  $Revision: 1.2 $
 * \author Ilaria Segoni - CERN
 */

#include <EventFilter/RPCRawToDigi/interface/RPCBXData.h>
#include <EventFilter/RPCRawToDigi/interface/RMBErrorData.h>
#include <EventFilter/RPCRawToDigi/interface/RPCChannelData.h>
#include <EventFilter/RPCRawToDigi/interface/RPCChamberData.h>

#include <vector>

using namespace std;


class RPCEventData{

public:
   RPCEventData(){DCCDiscarded=0;}
   ~RPCEventData(){}

   /// Insert unpacked information
   void addBXData(RPCBXData & bx);  
   void addChnData(RPCChannelData & chn);  
   void addRPCChamberData(RPCChamberData & chmb);  
   void addRMBDiscarded(RMBErrorData & errorData);  
   void addRMBCorrupted(RMBErrorData & errorData);  
   void addDCCDiscarded();  
   
   /// Access Methods to  unpacked information
   vector<RPCBXData> bxData();
   vector<RPCChannelData>  dataChannel();
   vector<RPCChamberData>  dataChamber();
   vector<RMBErrorData> dataRMBDiscarded();
   vector<RMBErrorData> dataRMBCorrupted();
   int dccDiscarded();

private:

   vector<RPCBXData>  BXData;
   vector<RPCChannelData> ChnData;
   vector<RPCChamberData>  ChmbData;
    
   vector<RMBErrorData> RMBDiscarded;
   vector<RMBErrorData> RMBCorrupted;
   int DCCDiscarded;

};


#endif
