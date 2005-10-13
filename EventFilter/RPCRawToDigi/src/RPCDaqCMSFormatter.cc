/**  
*
 */

#include "EventFilter/RPCRawToDigi/interface/RPCDaqCMSFormatter.h"
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/DaqData.h>

#include "EventFilter/RPCRawToDigi/interface/RPCFEDDataFormat.h"
#include "EventFilter/RPCRawToDigi/interface/RPCFEDHeaderFormat.h"
#include "EventFilter/RPCRawToDigi/interface/RPCFEDTrailerFormat.h"

#include <DataFormats/RPCDigis/interface/RPCDigi.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/RPCDigis/interface/RPCDigiCollection.h>
#include <iostream>

using namespace std;
using namespace raw;

RPCDaqCMSFormatter::RPCDaqCMSFormatter(){
 

 length=0;
 bytesread=0;
 
}

RPCDaqCMSFormatter::~RPCDaqCMSFormatter(){

}


void RPCDaqCMSFormatter::interpretRawData(const FEDRawData & fedData,
        			       RPCDigiCollection & digicollection){


 const unsigned char * buf = fedData.data();

 length = fedData.data();
 bytesread=0;

 while(bytesread<length){
	int datasize= headerUnpack( RPCDigiCollection & digicollection);
	if (datasize!=0)   payLoadUnpack(datasize, RPCDigiCollection & digicollection);   
	TrailerUnpack(RPCDigiCollection & digicollection);      
 }
}


void RPCDaqCMSFormatter::headerUnpack(RPCDigiCollection & digicollection){

      //RPC Header  
      int datasize=RPCFEDHeaderFormat::getSizeInBytes();

      checkMemory(length, bytesread, datasize);
      DaqData<RPCFEDHeaderFormat> phead(buf, datasize);

      buf+=datasize;
      bytesread+=datasize;

       return datasize;

}

void RPCDaqCMSFormatter::payLoadUnpack(int datasize,RPCDigiCollection & digicollection){

	cout << "DetUnit #" << "??"  << endl;

	checkMemory(length, bytesread, datasize);
	DaqData<RPCFEDDataFormat> data(buf, datasize);

		for(int i=0;i<data.Nobjects();i++) {
	  RPCDigi::PackedDigiType packed;	  
	 
	  RPCDigi digi(packed);

	}
	buf+=datasize;
	bytesread+=datasize;
      

}

void RPCDaqCMSFormatter::TrailerUnpack(RPCDigiCollection & digicollection){

      datasize=RPCFEDTrailerFormat::getSizeInBytes();
      checkMemory(length, bytesread, datasize);
      DaqData<RPCFEDTrailerFormat> ptrail(buf, datasize);
      buf+=datasize;
      bytesread+=datasize;
      
}

