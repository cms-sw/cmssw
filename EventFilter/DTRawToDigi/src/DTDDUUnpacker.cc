/** \file
 *
 *  $Date: 2005/11/23 11:17:15 $
 *  $Revision: 1.3 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <EventFilter/DTRawToDigi/src/DTDDUUnpacker.h>
#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTROS25Unpacker.h>

#include <iostream>

using namespace std;

DTDDUUnpacker::DTDDUUnpacker() : ros25Unpacker(new DTROS25Unpacker) {
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
  FEDTrailer dduTrailer(reinterpret_cast<const unsigned char*>(index) + datasize - SLINK_WORD_SIZE);

  // Check Status Words
  DTDDUFirstStatusWord dduStatusWord1(index + datasize - 3*SLINK_WORD_SIZE);
  
  DTDDUSecondStatusWord dduStatusWord2(index + datasize - 2*SLINK_WORD_SIZE);

  
  //---- ROS data

  // Set the index to start looping on ROS data
  index += SLINK_WORD_SIZE;
  datasize -= 4*SLINK_WORD_SIZE; // header, trailer, 2 status words

  ros25Unpacker->interpretRawData(index, datasize, dduID, mapping, product);
  
}
