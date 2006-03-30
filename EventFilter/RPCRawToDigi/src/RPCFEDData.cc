/** \file
 * Implementation of class RPCFEDData
 *
 *  $Date: 2005/12/15 17:49:02 $
 *  $Revision: 1.1 $
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

 std::map<int, std::vector<RPCLinkBoardData > >channel_lb_map;
 std::vector<RPCLinkBoardData > data;

 if( rmbDataMap.find(rmb) != rmbDataMap.end() ){
	channel_lb_map = rmbDataMap[rmb];
	if(channel_lb_map.find(chn) != channel_lb_map.end()){
    		edm::LogInfo ("RPCUnpacker")<<"sonoqui2 ";
		data=channel_lb_map[chn];
	}  
	   
 }

 data.push_back( lbData );
 channel_lb_map[chn]=data;
 rmbDataMap[rmb]= channel_lb_map;
}   
      


void RPCFEDData::addRMBDiscarded(int rmb, int chn){
      std::vector<int> badChannels;
      if(  RMBDiscarded.find(rmb)!= RMBDiscarded.end()){
      		 badChannels= RMBDiscarded[rmb];
      }else{
		 RMBDiscarded[rmb]= badChannels;
      }
     
       badChannels.push_back( chn );
}  

  
void RPCFEDData::addRMBCorrupted(int rmb, int chn){
      std::vector<int> badChannels;
      if(  RMBCorrupted.find(rmb)!= RMBCorrupted.end()){
      		 badChannels= RMBCorrupted[rmb];
      }else{
		 RMBCorrupted[rmb]= badChannels;
      }
     
       badChannels.push_back( chn );
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

















