/** \file
 *
 *  $Date: 2005/11/21 17:38:48 $
 *  $Revision: 1.2 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTROS8Unpacker.h>

#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>
#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>

using namespace std;
using namespace edm;

#include <iostream>

#define SLINK_WORD_SIZE 8


void DTROS8Unpacker::interpretRawData(const unsigned char* index, int datasize,
				      edm::ESHandle<DTReadOutMapping>& mapping, 
				      std::auto_ptr<DTDigiCollection>& product) {


  /// CopyAndPaste from P.Ronchese unpacker

  const int wordLength = 4;
  int nrb = datasize / wordLength;
  int rob = 0;
  int ros = 0;
  int mbs[8];
  int lbs[8];
  int err[8];
  int itdc=0;
  int ievt=0;
  int ibwc=0;

  // Loop over the ROS8 words
  for ( int irb = 1; irb < nrb; irb++ ) {

    // The word
    int word = index[irb];

    // The word type
    int type = ( word >> 28 ) & 0xF;

    // ROS Header ??
    if ( type == 15 ) {
      rob =   word        & 0x7;
      ros = ( word >> 3 ) & 0xFF;
    } 

    // ROB Header ??
    else if ( type == 14 ) {
      int cmap =   word &        0xFF;
      int lock = ( word >> 8 ) & 0xFF;
      for ( int bit = 0; bit < 8; bit++ ) {
        mbs[bit] = cmap & 1;
        cmap >>= 1;
        lbs[bit] = lock & 1;
        lock >>= 1; 
        err[bit] = mbs[bit] && lbs[bit];
      }
    }

    // TDC Header/Trailer
    else if ( type <= 3 ) {
      itdc = ( word >> 24 ) & 0xF;
      ievt = ( word >> 12 ) & 0xFFF;
      ibwc =   word &         0xFFF; 
    }

    // TDC Measurement
    // Note that this is assumed to be reached after all previous types
    // have already been found...
    else if ( type >= 4 && type <= 5 ) {
      int itdc1 = ( word >> 24 ) & 0xF;
      int icha = ( word >> 19 ) & 0x1F;
      int time =   word         & 0x7FFFF;

      // int edge = ( type == 4 ? 0 : 1 );
      time >>= 2;

      // Map the RO channel to the DetId and wire
      DTDetId detId; 
      int dduID = 1;
      detId = mapping->readOutToGeometry(dduID, ros, rob, itdc1, icha);
      int wire = detId.wire();
      
      // Produce the digi
      DTDigi digi(wire, time);
      product->insertDigi(detId.layerId(),digi);
    }
  }
}
