/** \file
 *
 *  $Date: 2006/04/10 12:20:40 $
 *  $Revision: 1.8 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/interface/DTControlData.h>

#include <EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <EventFilter/DTRawToDigi/src/DTDDUUnpacker.h>
#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>

#include <iostream>

using namespace std;

DTDDUUnpacker::DTDDUUnpacker(const edm::ParameterSet& ps) : pset(ps) { 
  
  ros25Unpacker = new DTROS25Unpacker(ps);

  if(pset.getUntrackedParameter<bool>("performDataIntegrityMonitor",false)){
    cout<<"[DTDDUUnpacker]: Enabling Data Integrity Checks"<<endl;
    dataMonitor = edm::Service<DTDataMonitorInterface>().operator->(); 
  }

  debug = pset.getUntrackedParameter<bool>("debugMode",false);

}

DTDDUUnpacker::~DTDDUUnpacker() {
  cout<<"[DTDDUUnpacker]: Destructor"<<endl;
  delete ros25Unpacker;
  
}


void DTDDUUnpacker::interpretRawData(const unsigned int* index32, int datasize,
				     int dduID,
				     edm::ESHandle<DTReadOutMapping>& mapping, 
				     std::auto_ptr<DTDigiCollection>& product,
				     uint16_t rosList) {

  // Definitions
  const int wordSize_32 = 4;
  const int wordSize_64 = 8;

  int numberOf32Words = datasize/wordSize_32;
  int numberOf64Words = datasize/wordSize_64;

  const unsigned char* index8 = reinterpret_cast<const unsigned char*>(index32);


  //////////////////////
  /*  D D U   d a t a */
  //////////////////////

  // DDU header
  FEDHeader dduHeader(index8);
  if (debug)  cout<<"[DTDDUUnpacker]: FED Header, BXID:"<<dduHeader.bxID()
		  <<" L1ID "<<dduHeader.lvl1ID()<<endl;

  // DDU trailer
  // [BITS] stop before FED trailer := 8 bytes
  FEDTrailer dduTrailer(index8 + datasize - 1*wordSize_64); 
  if (debug)  cout<<"[DTDDUUnpacker]: FED Trailer, lenght:"<<dduTrailer.lenght()<<endl;

  // Control DDU data
  DTDDUData controlData(dduHeader,dduTrailer);

  // Check Status Words 
  vector<DTDDUFirstStatusWord> rosStatusWords;
  // [BITS] 3 words of 8 bytes + "rosId" bytes
  for (int rosId = 0; rosId < 12; rosId++ ) {
    int wordIndex8 = numberOf32Words*wordSize_32 - 3*wordSize_64 + rosId; 
    controlData.addROSStatusWord(DTDDUFirstStatusWord(index8[wordIndex8]));
  }
  
  // [BITS] 2 words of 8 bytes + 4 bytes (half 64 bit word)
  DTDDUSecondStatusWord dduStatusWord(index32[numberOf32Words - 2*wordSize_64/wordSize_32 + 1]);
  controlData.addDDUStatusWord(dduStatusWord);

  // Perform dqm if requested
//   if (pset.getUntrackedParameter<bool>("performDataIntegrityMonitor",false)) 
//     dataMonitor->processFED(controlData, ddu);  


  //////////////////////
  /*  R O S   d a t a */
  //////////////////////

  // Set the index to start looping on ROS data
  // [BITS] one 8 bytes word and one 4 bytes
  index32 += (wordSize_64+wordSize_32)/wordSize_32; 

  // Set the datasize to look only at ROS data 
  // [BITS] header, trailer, 2 status words
  datasize -= 4*wordSize_64; 

  // unpacking the ROS payload
  ros25Unpacker->interpretRawData(index32, datasize, dduID, mapping, product,dduStatusWord.rosList());

}
