/** \file
 *
 *  Implementation of RPCChamberData
 *
 *  $Date: 2005/11/07 15:41:52 $
 *  $Revision: 1.4 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCChamberData.h>

RPCChamberData::RPCChamberData(const unsigned int* index): 
    word_(index) {
    
    partitionData_= (*word_>>PARTITION_DATA_SHIFT)&PARTITION_DATA_MASK;
    halfP_ = (*word_ >> HALFP_SHIFT ) & HALFP_MASK;
    eod_ = (*word_ >> EOD_SHIFT ) & EOD_MASK;
    partitionNumber_ = (*word_ >> PARTITION_NUMBER_SHIFT ) & PARTITION_NUMBER_MASK;
    chamberNumber_ = (*word_ >> CHAMBER_SHIFT ) & CHAMBER_MASK ;
}


int RPCChamberData::partitionData(){
  
  return partitionData_;

}

int RPCChamberData::halfP(){
  
  return halfP_;

}

int RPCChamberData::eod(){
  
  return eod_;

}

int RPCChamberData::partitionNumber(){
  
  return partitionNumber_;

}

int RPCChamberData::chamberNumber(){
  
  return chamberNumber_;

}
