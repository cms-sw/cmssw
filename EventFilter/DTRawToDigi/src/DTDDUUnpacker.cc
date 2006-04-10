/** \file
 *
 *  $Date: 2006/04/07 15:36:04 $
 *  $Revision: 1.7 $
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

}

DTDDUUnpacker::~DTDDUUnpacker() {
  delete ros25Unpacker;
  
}


void DTDDUUnpacker::interpretRawData(const unsigned int* index, int datasize,
				     int dduID,
				     edm::ESHandle<DTReadOutMapping>& mapping, 
				     std::auto_ptr<DTDigiCollection>& product,
				     uint16_t rosList) {



  // DDU header
  FEDHeader dduHeader(reinterpret_cast<const unsigned char*>(index));

  // DDU trailer
  // [BITS] stop before FED trailer := 8 bytes
  FEDTrailer dduTrailer(reinterpret_cast<const unsigned char*>(index) + datasize - 8); 
  
  // Control DDU data
  DTDDUData controlData(dduHeader,dduTrailer);

  // Check Status Words 
  vector<DTDDUFirstStatusWord> rosStatusWords;
  // [BITS] 3 words of 8 bytes + "rosId" bytes
  for (int rosId = 0; rosId < 12; rosId++ ) {
    controlData.addROSStatusWord(DTDDUFirstStatusWord(index[datasize - 3*8 + rosId]));
  }
  
  // [BITS] 2 words of 8 bytes + 4 bytes (half 64 bit word)
  DTDDUSecondStatusWord dduStatusWord(index[datasize - 2*8 + 4]);
  controlData.addDDUStatusWord(dduStatusWord);

  //---- ROS data

  // Set the index to start looping on ROS data
  index += 1*2;
  // Set the datasize to look only at ROS data 
  datasize -= 4*2; // header, trailer, 2 status words

  ros25Unpacker->interpretRawData(index, datasize, dduID, mapping, product,dduStatusWord.rosList());

  
  // Perform dqm if requested
  if (pset.getUntrackedParameter<bool>("performDataIntegrityMonitor",false)) 
    dataMonitor->processFED(controlData);  
  
}
