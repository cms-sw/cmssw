/** \file
 *
 *  Implementation of RPCLinkBoardData
 *
 *  $Date: 2006/05/03 16:36:22 $
 *  $Revision: 1.2 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h>

RPCLinkBoardData::RPCLinkBoardData(): 
halfP_(0),eod_(0),partitionNumber_(0),lbNumber_(0){
 	bitsOn_.clear();
}

RPCLinkBoardData::RPCLinkBoardData(std::vector<int> bits, int halfP, int eod, int partitionNumber, int lbNumber): 
halfP_(halfP),eod_(eod),partitionNumber_(partitionNumber),lbNumber_(lbNumber){
 	for(std::vector<int>::iterator itr= bits.begin(); itr!= bits.end(); ++itr){
		bitsOn_.push_back(*itr);
	}
}

void  RPCLinkBoardData::setBits(std::vector<int> bits){bitsOn_=bits;}

void  RPCLinkBoardData::setHalfP(int hp){halfP_=hp;}

void  RPCLinkBoardData::setEod(int eod){eod_=eod;}

void  RPCLinkBoardData::setPartitionNumber(int partNumb){partitionNumber_=partNumb;}

void  RPCLinkBoardData::setLbNumber(int lbNumb){lbNumber_=lbNumb;}

int RPCLinkBoardData::lbData() const{
////AK Temporary 
  const int BITS_PER_PARTITION=8;
/////
  int partitionData = 0; 
  for (std::vector<int>::const_iterator iv = bitsOn_.begin(); iv != bitsOn_.end(); iv++ ) {
    int ibit = (partitionNumber_)? (*iv)%(partitionNumber_*BITS_PER_PARTITION) : (*iv);
    partitionData |= (1<<ibit); 
  }

  return  partitionData;
}

std::vector<int> RPCLinkBoardData::bitsOn() const{  
	return bitsOn_ ;
}

int RPCLinkBoardData::halfP() const{  
	return halfP_;
}

int RPCLinkBoardData::eod() const{  
	return eod_;
}

int RPCLinkBoardData::partitionNumber() const{  
	return partitionNumber_;
}

int RPCLinkBoardData::lbNumber() const{
	return lbNumber_;
}
