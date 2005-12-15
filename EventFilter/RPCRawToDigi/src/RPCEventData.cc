/** \file
 * Implementation of class RPCEventData
 *
 *  $Date: 2005/12/14 13:35:22 $
 *  $Revision: 1.7 $
 *
 * \author Ilaria Segoni
 */
 
#include <EventFilter/RPCRawToDigi/interface/RPCEventData.h>

 
void RPCEventData::addBXData(RPCBXData & bx){
 BXData.push_back(bx);
}   



void RPCEventData::addChnData(RPCChannelData & chn){
  ChnData.push_back(chn);
}  


void RPCEventData::addRPCChamberData(RPCChamberData & chmb){
  ChmbData.push_back(chmb);
}


void RPCEventData::addRMBDiscarded(RMBErrorData & errorData){
   RMBDiscarded.push_back(errorData);
}  

  
void RPCEventData::addRMBCorrupted(RMBErrorData & errorData){
    RMBCorrupted.push_back(errorData);
}   


void RPCEventData::addDCCDiscarded(){
     DCCDiscarded++;
}  



vector<RPCBXData> RPCEventData::bxData(){
    return BXData;
}


vector<RPCChannelData>  RPCEventData::dataChannel(){
    return ChnData;
}


vector<RPCChamberData>  RPCEventData::dataChamber(){
    return ChmbData;
}



vector<RMBErrorData> RPCEventData::dataRMBDiscarded(){
    return  RMBDiscarded;
}


vector<RMBErrorData> RPCEventData::dataRMBCorrupted(){
    return  RMBCorrupted;
}


int RPCEventData::dccDiscarded(){
    return DCCDiscarded;
}

















