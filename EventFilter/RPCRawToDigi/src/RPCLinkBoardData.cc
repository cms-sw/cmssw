/** \file
 *
 *  Implementation of RPCLinkBoardData
 *
 *  $Date: 2005/12/12 17:30:58 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
 */


#include <EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h>

RPCLinkBoardData::RPCLinkBoardData(): 
halfP_(0),eod_(0),partitionNumber_(0),lbNumber_(0){
 	bitsOn_.clear();
}

void  RPCLinkBoardData::setBits(std::vector<int> bits){bitsOn_=bits;}
void  RPCLinkBoardData::setHalfP(int hp){halfP_=hp;}
void  RPCLinkBoardData::setEod(int eod){eod_=eod;}
void  RPCLinkBoardData::setPartitionNumber(int partNumb){partitionNumber_=partNumb;}
void  RPCLinkBoardData::setLbNumber(int lbNumb){lbNumber_=lbNumb;}


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
