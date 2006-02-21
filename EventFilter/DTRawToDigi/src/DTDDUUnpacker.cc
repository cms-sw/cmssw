/** \file
 *
 *  $Date: 2006/02/14 17:06:19 $
 *  $Revision: 1.5 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <EventFilter/DTRawToDigi/src/DTDDUUnpacker.h>
#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>

#include <iostream>

using namespace std;

DTDDUUnpacker::DTDDUUnpacker(const edm::ParameterSet& ps) : pset(ps) { 
  
  ros25Unpacker = new DTROS25Unpacker(ps);
}

DTDDUUnpacker::~DTDDUUnpacker() {
  delete ros25Unpacker;
}


void DTDDUUnpacker::interpretRawData(const unsigned int* index, int datasize,
				     int dduID,
				     edm::ESHandle<DTReadOutMapping>& mapping, 
				     std::auto_ptr<DTDigiCollection>& product) {

  // Check DDU header
  FEDHeader dduHeader(reinterpret_cast<const unsigned char*>(index));

  // Check DDU trailer
  FEDTrailer dduTrailer(reinterpret_cast<const unsigned char*>(index) + datasize - 1*2);

  // Check Status Words (CHECK THIS)
  DTDDUFirstStatusWord dduStatusWord1(index[datasize - 3*2]);
  
  DTDDUSecondStatusWord dduStatusWord2(index[datasize - 2*2]);

  
  //---- ROS data

  // Set the index to start looping on ROS data
  index += 1*2;
  datasize -= 4*2; // header, trailer, 2 status words

  ros25Unpacker->interpretRawData(index, datasize, dduID, mapping, product);
  
}
