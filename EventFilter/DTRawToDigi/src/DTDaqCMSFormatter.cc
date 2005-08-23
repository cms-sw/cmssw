/**  
 *  See header file for a description of this class.
 *  
 * 
 *
 *  $Date: 2005/07/13 09:06:50 $
 *  $Revision: 1.1 $
 *  \author G. Bruno - CERN, EP Division
 */

#include "DTDaqCMSFormatter.h"
//#include "CommonDet/DetReadout/interface/FrontEndDriver.h"
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/DaqData.h>
#include "DTFEDDataFormat.h"
#include "DTFEDHeaderFormat.h"
#include "DTFEDTrailerFormat.h"
//#include "Utilities/GenUtil/interface/ioutils.h"
//#include "Muon/MBDetector/interface/MuBarBaseReadout.h"
//#include "Muon/MBDetector/interface/MuBarLayer.h"
//#include "Utilities/Notification/interface/Verbose.h"

#include <DataFormats/DTDigis/interface/DTDigi.h>
#include <DataFormats/MuonDetId/interface/DTDetId.h>
#include <DataFormats/DTDigis/interface/DTDigiCollection.h>
#include <iostream>

using namespace std;
using namespace raw;

void DTDaqCMSFormatter::interpretRawData(const FEDRawData & fedData,
					 DTDigiCollection& digicollection){

  cout <<"MBDaqCMSFormatter::interpretRawData() enter" <<  endl;

//   vector<MuBarBaseReadout*> readouts;
//   for(vector<DetUnit *>::const_iterator it=fed->detsBegin(); it!=fed->detsEnd(); it++) readouts.push_back(dynamic_cast<MuBarBaseReadout *>(&((*it)->readout())));

  const unsigned char * buf = fedData.data();
  int length = fedData.data_.size();
  int bytesread=0;
  
  try{

    //    for(unsigned int idf=0; idf<readouts.size(); idf++){
    while(bytesread<length){

      //MB Header  
      int datasize=DTFEDHeaderFormat::getSizeInBytes();

      checkMemory(length, bytesread, datasize);
      DaqData<DTFEDHeaderFormat> phead(buf, datasize);
      buf+=datasize;
      bytesread+=datasize;

      //FED Data
      int readoutindex=phead.getValue(0);
      datasize=phead.getValue(1);

//       if (readoutindex <0 || readoutindex >= readouts.size())  throw string("MBDaqCmsFormatter:InterpretRawData() : ERROR. readout index (" + string(toa()(readoutindex)) + ") out of allowed range: [0," + string(toa()(readouts.size()-1)) + "]");

      if (datasize!=0){

// 	if (debugV) {
// 	  MuBarLayer * det = dynamic_cast<MuBarLayer *>(&(readouts[readoutindex]->det())); 
// 	  cout<<"Going to fill digi cache of DetUnit at "<<det->position()<<endl;
// 	}
	
	cout << "DetUnit #" << readoutindex << endl;

	checkMemory(length, bytesread, datasize);
	DaqData<DTFEDDataFormat> data(buf, datasize);
	//fill slave
	for(int i=0;i<data.Nobjects();i++) {
	  DTDigi::PackedDigiType packed;	  
	  packed.wire      = data.getValue(0,i);
	  packed.layer     = data.getValue(1,i);
	  packed.slayer    = data.getValue(2,i);
	  packed.number    = data.getValue(3,i);
	  packed.trailer1  = data.getValue(4,i);
	  packed.counts    = data.getValue(5,i);
	  packed.trailer2  = data.getValue(6,i);
	  DTDigi digi(packed);
	  //	  readouts[readoutindex]->add(digi);

#warning Fake mapping of layer !

	  DTDetId   layer(1,               //wheel
			  readoutindex,    //station
			  readoutindex *2, //sector
			  readoutindex *3, //slayer
			  readoutindex *4);//layer


          digicollection.insertDigi(layer,digi);

	  cout << " New DT digi: ";
	  digi.print();
	}
	buf+=datasize;
	bytesread+=datasize;
      }
      
      //MB FED Trailer  
      datasize=DTFEDTrailerFormat::getSizeInBytes();
      checkMemory(length, bytesread, datasize);
      DaqData<DTFEDTrailerFormat> ptrail(buf, datasize);
      buf+=datasize;
      bytesread+=datasize;
      
    }

  }
  catch (string s){
    cout<<"MBDaqCMSFormatter - Exception caught: " << s <<endl;
    cout<<"Interpretation of compressed data failed!"<<endl;   
  }

//   for(int idf = 0; idf != readouts.size(); idf++){
//     readouts[idf]->setDigisUpToDate();
//   }
  


}


// DaqFEDRawData * MBDaqCMSFormatter::formatData(FrontEndDriver * fed){


//   if ( debugV )  cout <<"MBDaqCmsFormatter::formatData() enter" <<  endl;

//   vector< MuBarBaseReadout *>  readouts;
//   for(vector<DetUnit *>::const_iterator it=fed->detsBegin(); it!=fed->detsEnd(); it++) readouts.push_back(dynamic_cast<MuBarBaseReadout *>(&((*it)->readout())));


//   int NDets=readouts.size();
//   int digisize=0;

//   for(int idf=0; idf<NDets; idf++) {
//     digisize+= MBFEDDataFormat::getSizeInBytes(readouts[idf]->ndigis());
//   }

//   int fedDataSize= NDets* ( MBFEDHeaderFormat::getSizeInBytes()+
// 			    MBFEDTrailerFormat::getSizeInBytes() ) + digisize;

//   DaqFEDRawData * fedData = new DaqFEDRawData(fedDataSize);

//   int filledmemory=0;
//   char * buf = fedData->data();


//   try {

//     for(unsigned int idf=0; idf<NDets; idf++) {

//       //MB FED Header
//       vector<unsigned int> pheadentry;
//       pheadentry.push_back(idf);
//       int numDigis = readouts[idf]->ndigis();

//       int datasize=MBFEDDataFormat::getSizeInBytes(numDigis);
//       pheadentry.push_back((unsigned int)datasize);

//       DaqData<MBFEDHeaderFormat> phead(pheadentry);

//       checkMemory(fedDataSize, filledmemory, phead.Size());
//       memmove(buf, phead.Buffer(), phead.Size());
//       buf+=phead.Size();
//       filledmemory+=phead.Size();

//       // FED data

//       if(datasize){
// 	if (debugV) {
// 	  MuBarLayer * det = dynamic_cast<MuBarLayer *>(&(readouts[idf]->det())); 
// 	  cout<<"Found non empty DetUnit at "<<det->position()<<endl;
// 	}

// 	vector<unsigned int> feddigis;
// 	feddigis.reserve(numDigis * MBFEDDataFormat::getNumberOfFields());

// 	for(vector<MuBarDigi>::const_iterator digiit = readouts[idf]->begin() ; digiit != readouts[idf]->end(); digiit++) {
// 	  if (debugV) (*((vector<MuBarDigi>::const_iterator)(digiit))).print();
// 	  MuBarDigi::PackedDigiType data = (*digiit).packedData();
// 	  feddigis.push_back(data.wire     );
// 	  feddigis.push_back(data.layer    );
// 	  feddigis.push_back(data.slayer   );
// 	  feddigis.push_back(data.number   );
// 	  feddigis.push_back(data.trailer1 );
// 	  feddigis.push_back(data.counts   );
// 	  feddigis.push_back(data.trailer2 );
// 	}

// 	DaqData<MBFEDDataFormat> fdata(feddigis);
      
// 	checkMemory(fedDataSize, filledmemory, fdata.Size());
// 	memmove(buf, fdata.Buffer(), fdata.Size());
// 	buf+=fdata.Size();
// 	filledmemory+=fdata.Size();
//       }

//       //MB FED Trailer
//       vector<unsigned int> ptrailentry;
//       ptrailentry.push_back(0); //Not yet decided what to put
//       DaqData<MBFEDTrailerFormat> ptrail(ptrailentry);
      
//       checkMemory(fedDataSize, filledmemory, ptrail.Size());
//       memmove(buf, ptrail.Buffer(), ptrail.Size());
//       buf+=ptrail.Size();
//       filledmemory+=ptrail.Size();

//     }     

//     if (filledmemory < fedDataSize) throw string("MBDaqCMSFormatter::formatData() : ERROR. the memory that has been filled is less than the reserved one");

//   }

//   catch (string s){
//     cout<<"MBDaqCMSFormatter - Exception caught: " << s <<endl;
//     cout<<"Data compression failed! Returning null pointer"<<endl;   
//     delete fedData;
//     return NULL;
//   }

//   return fedData;

// }



