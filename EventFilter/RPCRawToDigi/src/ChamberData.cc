/** \file
 *
 *  $Date: 2005/10/21 10:58:41 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/ChamberData.h>

ChamberData::ChamberData(const unsigned char* index){

       word_(reinterpret_cast<const unsigned int*>(index)) ;

        partitionData_   = (*word_ >> PARTITION_DATA_SHIFT   ) & PARTITION_DATA_MASK ;
	halfP_           = (*word_ >> HALFP_SHIFT            ) & HALFP_MASK;
	eod_             = (*word_ >> EOD_SHIFT              ) & EOD_MASK;
	partitionNumber_ = (*word_ >> PARTITION_NUMBER_SHIFT ) & PARTITION_NUMBER_MASK;
	chamberNumber_   = (*word_ >> CHAMBER_SHIFT          ) & CHAMBER_MASK ;
}





int ChamberData::partitionData(){
  
  return partitionData_;

}

int ChamberData::halfP(){
  
  return halfP_;

}

int ChamberData::eod(){
  
  return eod_;

}

int ChamberData::partitionNumber(){
  
  return partitionNumber_;

}

int ChamberData::chamberNumber(){
  
  return chamberNumber_;

}
