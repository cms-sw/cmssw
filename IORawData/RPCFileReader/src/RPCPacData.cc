
#include "IORawData/RPCFileReader/interface/RPCPacData.h"
#include <iomanip>
#include <sstream>
#include <iostream>

using namespace std;

RPCPacData::RPCPacData() {		
  partitionData_  = 0;
  partitionNum_   = 0;
  partitionDelay_ = 0;
  endOfData_      = 0;
  halfPartition_  = 0;
  lbNum_          = 0;	
}

RPCPacData::RPCPacData(unsigned int rawData) {		
  unsigned int shift = 0;
  
  partitionData_  =  rawData &    0xff          ; shift += 8;
  partitionNum_   = (rawData &   0xf00) >> shift; shift += 4;
  partitionDelay_ = (rawData &  0x7000) >> shift; shift += 3;
  endOfData_      = (rawData &  0x8000) >> shift; shift += 1;
  halfPartition_  = (rawData & 0x10000) >> shift; shift += 1;
  lbNum_          = (rawData & 0x60000) >> shift; shift += 2;	
  
}

RPCPacData::RPCPacData(unsigned int partData,
		       unsigned int partNo, 
		       unsigned int partDelay,
		       unsigned int eofData,
		       unsigned int halfPart,
		       unsigned int lbNo)  {	
  partitionData_  =  partData;
  partitionNum_   =  partNo;
  partitionDelay_ =  partDelay;
  endOfData_      =  eofData;
  halfPartition_  =  halfPart;
  lbNum_          =  lbNo;	
}

unsigned int RPCPacData::toRaw() {		
  unsigned int rawData = 0;
  unsigned int shift = 0;

  rawData = partitionData_                                 ; shift += 8;
  rawData = rawData | ((partitionNum_  << shift) &   0xf00); shift += 4;
  rawData = rawData | ((partitionDelay_<< shift )&  0x7000); shift += 3;
  rawData = rawData | ((endOfData_     << shift )&  0x8000); shift += 1;
  rawData = rawData | ((halfPartition_ << shift )& 0x10000); shift += 1;
  rawData = rawData | ((lbNum_         << shift )& 0x60000); shift += 2;

  return rawData;
}

bool RPCPacData::operator == (const RPCPacData& right) const {
  if( partitionData_  == right.partitionData()    &&
      partitionNum_   == right.partitionNum()     &&
      partitionDelay_ == right.partitionDelay()   &&
      endOfData_      == right.endOfData()        &&
      halfPartition_  == right.halfPartition()    &&
      lbNum_          == right.lbNum()	            )
    return true;
  else
    return false;
}
	
bool RPCPacData::operator != (const RPCPacData& right) const {
  return !(*this == right);
}

string RPCPacData::toString() {
  ostringstream ostr;
  
  ostr << lbNum_ << " "
       << halfPartition_ << " "
       << endOfData_ << " " 
       << partitionDelay_<< " " 
       << setw(2)<< dec 
       << partitionNum_ << " "
       << setw(2)<< hex << setfill('0')
       << partitionData_;
  
  return ostr.str();
}
