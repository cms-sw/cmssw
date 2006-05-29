/** \file
 * Implementation of class RPCFEDData
 *
 *  $Date: 2006/05/04 09:39:48 $
 *  $Revision: 1.2 $
 *
 * \author Ilaria Segoni
 */
 
#include <EventFilter/RPCRawToDigi/interface/RPCFEDData.h>
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

















