/** \file
 * Implementation of class RPCFEDData
 *
 *  $Date: 2006/10/08 12:11:12 $
 *  $Revision: 1.4 $
 *
 * \author Ilaria Segoni
 */
 
#include "EventFilter/RPCRawToDigi/interface/RPCFEDData.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void RPCFEDData::addCdfHeader(FEDHeader & header){
	cdfHeaders.push_back(header);
}   

void RPCFEDData::addCdfTrailer(FEDTrailer & trailer){
	cdfTrailers.push_back(trailer);
}   

 
void RPCFEDData::addBXData(int bx){
	bxCounts.push_back(bx);
}   

void RPCFEDData::addRMBData(int rmb,int chn, RPCLinkBoardData lbData){

 std::map<int, std::vector<RPCLinkBoardData > >tbLinkInputNumber_lb_map;
 std::vector<RPCLinkBoardData > data;

 if( rmbDataMap.find(rmb) != rmbDataMap.end() ){
	tbLinkInputNumber_lb_map = rmbDataMap[rmb];
	if(tbLinkInputNumber_lb_map.find(chn) != tbLinkInputNumber_lb_map.end()){
		data=tbLinkInputNumber_lb_map[chn];
	}  
	   
 }

 data.push_back( lbData );
 tbLinkInputNumber_lb_map[chn]=data;
 rmbDataMap[rmb]= tbLinkInputNumber_lb_map;
}   
      


void RPCFEDData::addRMBDiscarded(int rmb, int chn){
      std::vector<int> badTbLinkInputNumbers;
      if(  RMBDiscarded.find(rmb)!= RMBDiscarded.end()){
      		 badTbLinkInputNumbers= RMBDiscarded[rmb];
      }else{
		 RMBDiscarded[rmb]= badTbLinkInputNumbers;
      }
     
       badTbLinkInputNumbers.push_back( chn );
}  

  
void RPCFEDData::addRMBCorrupted(int rmb, int chn){
      std::vector<int> badTbLinkInputNumbers;
      if(  RMBCorrupted.find(rmb)!= RMBCorrupted.end()){
      		 badTbLinkInputNumbers= RMBCorrupted[rmb];
      }else{
		 RMBCorrupted[rmb]= badTbLinkInputNumbers;
      }
     
       badTbLinkInputNumbers.push_back( chn );
}   

void RPCFEDData::addRMBDisabled(int rmbDisabled){     
      RMBDisabledList.push_back( rmbDisabled );
}   

void RPCFEDData::addDCCDiscarded(){
	DCCDiscarded++;
}  

std::vector<FEDHeader> RPCFEDData::fedHeaders() const{
	return cdfHeaders;
}
std::vector<FEDTrailer>  RPCFEDData::fedTrailers() const{
	return cdfTrailers;
}


std::vector<int> RPCFEDData::bxData() const{
	return bxCounts;
}

std::map<int , std::map<int, std::vector<RPCLinkBoardData> > > RPCFEDData::rmbData() const{
	return  rmbDataMap;
}


std::map<int,std::vector<int> > RPCFEDData::dataRMBDiscarded() const{
	return  RMBDiscarded;
}


std::map<int,std::vector<int> > RPCFEDData::dataRMBCorrupted() const{
	return  RMBCorrupted;
}


int RPCFEDData::dccDiscarded() const{
	return DCCDiscarded;
}


/*

// THE REST IS REMNANT OF ILARIA CODE WHICH IS NOT YET PORTED.
      namespace bits{
            static const int BITS_PER_PARTITION=8;
      }
      namespace error{
            static const int TB_LINK_MASK  = 0X1F;
            static const int TB_LINK_SHIFT =0;

            static const int TB_RMB_MASK = 0X3F;
            static const int TB_RMB_SHIFT =5;

            static const int RMB_DISABLED_MASK = 0X6;
            static const int RMB_DISABLED_SHIFT =0;
      }

    
    if(typeOfRecord==RPCRecord::RMBDiscarded || typeOfRecord==RPCRecord::RMBCorrupted ) this->unpackRMBCorruptedRecord(recordIndexInt,typeOfRecord,rawData);
    if(typeOfRecord==RPCRecord::RMBDisabled ) this->unpackRMBDisabledRecord(recordIndexInt,typeOfRecord,rawData);
    if(typeOfRecord==RPCRecord::DCCDiscarded) rawData.addDCCDiscarded();

void RPCRecordFormatter::unpackRMBCorruptedRecord(const unsigned int* recordIndexInt,enum RPCRecord::recordTypes type,RPCFEDData & rawData) {
    int tbLinkInputNumber = (* recordIndexInt>> rpcraw::error::TB_LINK_SHIFT )& rpcraw::error::TB_LINK_MASK;
    int tbRmb   = (* recordIndexInt>> rpcraw::error::TB_RMB_SHIFT)  & rpcraw::error::TB_RMB_MASK;
    if(type==RPCRecord::RMBDiscarded) rawData.addRMBDiscarded(tbRmb, tbLinkInputNumber);
    if(type==RPCRecord::RMBCorrupted) rawData.addRMBCorrupted(tbRmb, tbLinkInputNumber);  
 }


void RPCRecordFormatter::unpackRMBDisabledRecord(const unsigned int* recordIndexInt,enum RPCRecord::recordTypes type, RPCFEDData & rawData) {
    	int rmbDisabled = (* recordIndexInt>> rpcraw::error::RMB_DISABLED_SHIFT ) & rpcraw::error::RMB_DISABLED_MASK;
	rawData.addRMBDisabled(rmbDisabled);  
    	edm::LogInfo ("RPCUnpacker")<< "Found RMB Disabled: "<<rmbDisabled;
 }

*/












