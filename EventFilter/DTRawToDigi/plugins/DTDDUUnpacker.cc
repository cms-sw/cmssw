/** \file
 *
 *  $Date: 2010/03/08 08:57:41 $
 *  $Revision: 1.8 $
 *  \author  M. Zanetti - INFN Padova 
 * FRC 060906
 */

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>

#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/interface/DTControlData.h>

#include <EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <EventFilter/DTRawToDigi/plugins/DTDDUUnpacker.h>
#include <EventFilter/DTRawToDigi/plugins/DTROS25Unpacker.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using namespace std;
using namespace edm;

DTDDUUnpacker::DTDDUUnpacker(const edm::ParameterSet& ps) : dduPSet(ps) { 
  
  // the ROS unpacker
  ros25Unpacker = new DTROS25Unpacker(dduPSet.getParameter<edm::ParameterSet>("rosParameters"));
  
  // parameters
  localDAQ = dduPSet.getUntrackedParameter<bool>("localDAQ",false);
  performDataIntegrityMonitor = dduPSet.getUntrackedParameter<bool>("performDataIntegrityMonitor",false);
  debug = dduPSet.getUntrackedParameter<bool>("debug",false);

  // enable DQM if Service is available
  if(performDataIntegrityMonitor) {
    if (edm::Service<DTDataMonitorInterface>().isAvailable()) {
      dataMonitor = edm::Service<DTDataMonitorInterface>().operator->(); 
    } else {
      LogWarning("DTRawToDigi|DTDDUUnpacker") << 
	"[DTDDUUnpacker] WARNING! Data Integrity Monitoring requested but no DTDataMonitorInterface Service available" << endl;
      performDataIntegrityMonitor = false;
    }
  }

}


DTDDUUnpacker::~DTDDUUnpacker() {
  delete ros25Unpacker;
}


void DTDDUUnpacker::interpretRawData(const unsigned int* index32, int datasize,
				     int dduID,
				     edm::ESHandle<DTReadOutMapping>& mapping, 
				     std::auto_ptr<DTDigiCollection>& detectorProduct,
				     std::auto_ptr<DTLocalTriggerCollection>& triggerProduct,
				     uint16_t rosList) {

  // Definitions
  const int wordSize_32 = 4;
  const int wordSize_64 = 8;

  int numberOf32Words = datasize/wordSize_32;

  const unsigned char* index8 = reinterpret_cast<const unsigned char*>(index32);


  //////////////////////
  /*  D D U   d a t a */
  //////////////////////

  // DDU header
  FEDHeader dduHeader(index8);
  if (dduHeader.check()) {
    if(debug) cout << "[DTDDUUnpacker] FED Header. BXID: "<<dduHeader.bxID()
		   << " L1ID: "<<dduHeader.lvl1ID() <<endl;
  } else {
    LogWarning("DTRawToDigi|DTDDUUnpacker") << "[DTDDUUnpacker] WARNING!, this is not a DDU Header, FED ID: "
					    << dduID << endl;
  }

  // DDU trailer
  // [BITS] stop before FED trailer := 8 bytes
  FEDTrailer dduTrailer(index8 + datasize - 1*wordSize_64); 

  if (dduTrailer.check()) {
    if(debug) cout << "[DTDDUUnpacker] FED Trailer. Lenght of the DT event: "
		   << dduTrailer.lenght() << endl;
  } else {
    LogWarning("DTRawToDigi|DTDDUUnpacker") << "[DTDDUUnpacker] WARNING!, this is not a DDU Trailer, FED ID: "
					    << dduID << endl;
  }


  // Control DDU data
  DTDDUData controlData(dduHeader,dduTrailer);
  // check the CRC set in the FED trailer (FCRC errors)
  controlData.checkCRCBit(index8 + datasize - 1*wordSize_64);

  // Check Status Words 
  vector<DTDDUFirstStatusWord> rosStatusWords;
  // [BITS] 3 words of 8 bytes + "rosId" bytes
  // In the case we are reading from DMA, the status word are swapped as the ROS data
  if (localDAQ) {
    // DDU channels from 1 to 4
    for (int rosId = 0; rosId < 4; rosId++ ) {
      int wordIndex8 = numberOf32Words*wordSize_32 - 3*wordSize_64 + wordSize_32 + rosId; 
      controlData.addROSStatusWord(DTDDUFirstStatusWord(index8[wordIndex8]));
    }
    // DDU channels from 5 to 8
    for (int rosId = 0; rosId < 4; rosId++ ) {
      int wordIndex8 = numberOf32Words*wordSize_32 - 3*wordSize_64 + rosId; 
      controlData.addROSStatusWord(DTDDUFirstStatusWord(index8[wordIndex8]));
    }
    // DDU channels from 9 to 12
    for (int rosId = 0; rosId < 4; rosId++ ) {
      int wordIndex8 = numberOf32Words*wordSize_32 - 2*wordSize_64 + wordSize_32 + rosId; 
      controlData.addROSStatusWord(DTDDUFirstStatusWord(index8[wordIndex8]));
    }
  }
  else {
    for (int rosId = 0; rosId < 12; rosId++ ) {
      int wordIndex8 = numberOf32Words*wordSize_32 - 3*wordSize_64 + rosId; 
      controlData.addROSStatusWord(DTDDUFirstStatusWord(index8[wordIndex8]));
    }
  }

  int theROSList;
  // [BITS] 2 words of 8 bytes + 4 bytes (half 64 bit word)
  // In the case we are reading from DMA, the status word are swapped as the ROS data
  if (localDAQ) {
    DTDDUSecondStatusWord dduStatusWord(index32[numberOf32Words - 2*wordSize_64/wordSize_32]);
    controlData.addDDUStatusWord(dduStatusWord);
    theROSList =  dduStatusWord.rosList();
  }
  else {
    DTDDUSecondStatusWord dduStatusWord(index32[numberOf32Words - 2*wordSize_64/wordSize_32 + 1]);
    controlData.addDDUStatusWord(dduStatusWord);
    theROSList =  dduStatusWord.rosList();
  }


  //////////////////////
  /*  R O S   d a t a */
  //////////////////////

  // Set the index to start looping on ROS data
  // [BITS] one 8 bytes word
  index32 += (wordSize_64)/wordSize_32; 

  // Set the datasize to look only at ROS data 
  // [BITS] header, trailer, 2 status words
  datasize -= 4*wordSize_64; 

  // unpacking the ROS payload
  ros25Unpacker->interpretRawData(index32, datasize, dduID, mapping, detectorProduct, triggerProduct, theROSList);

  // Perform DQM if requested
  if (performDataIntegrityMonitor) 
    dataMonitor->processFED(controlData, ros25Unpacker->getROSsControlData(),dduID);  
  
}
