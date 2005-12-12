#ifndef RPCEventData_h
#define RPCEventData_h

/** \class RPCEventData
 * Container for Unpacked RPC Data, used by DQM module
 *
 *  $Date: 2005/11/24 18:06:54 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */

#include <EventFilter/RPCRawToDigi/interface/RPCBXData.h>
#include <EventFilter/RPCRawToDigi/interface/RMBErrorData.h>

#include <vector>


using namespace std;


class RPCEventData{

public:
   RPCEventData(){DCCDiscarded=0;}
   ~RPCEventData(){}

   void addBXData(RPCBXData & bxData){data.push_back(bxData);}  
   void addRMBDiscarded(RMBErrorData & errorData){RMBDiscarded.push_back(errorData);}  
   void addRMBCorrupted(RMBErrorData & errorData){RMBCorrupted.push_back(errorData);}  
   void addDCCDiscarded(){DCCDiscarded++;}  
   
   vector<RPCBXData> bxData(){return data;}
   vector<RMBErrorData> dataRMBDiscarded(){return  RMBDiscarded;}
   vector<RMBErrorData> dataRMBCorrupted(){return  RMBCorrupted;}
   int dccDiscarded(){return DCCDiscarded;}

private:

   vector<RPCBXData>  data;
   vector<RMBErrorData> RMBDiscarded;
   vector<RMBErrorData> RMBCorrupted;
   int DCCDiscarded;

};


#endif
