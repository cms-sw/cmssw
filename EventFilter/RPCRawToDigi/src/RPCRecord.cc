/** \file
 *
 *  $Date: 2005/10/21 16:45:41 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCRecord.h>
#include <EventFilter/RPCRawToDigi/interface/RPCControlRecord.h>

RPCRecord::RPCRecord(const unsigned char* index){

    word_(reinterpret_cast<const unsigned int*>(index)) {};

}



enum recordTypes RPCRecord::type() {
    
    enum recordTypes wordType = UndefinedType;
    
    // Chamber Data
      if ( ((*word_ & RECORD_TYPE_MASK) >> RECORD_TYPE_SHIFT) == chamberZeroFlag) wordType = DataChamber;
      if ( ((*word_ & RECORD_TYPE_MASK) >> RECORD_TYPE_SHIFT) == chamberOneFlag ) wordType = DataChamber;
      if ( ((*word_ & RECORD_TYPE_MASK) >> RECORD_TYPE_SHIFT) == chamberTwoFlag ) wordType = DataChamber;

    // Control Word
      if ( ((*word_ & RECORD_TYPE_MASK) >> RECORD_TYPE_SHIFT) == controlWordFlag) wordType = Control;

    return wordType;
}


const unsigned char* RPCRecord::next() 
{ 
	word_ += RPC_RECORD_BIT_SIZE; 
	return word_;
}

