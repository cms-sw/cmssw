/** \file
 *
 *  $Date: 2005/11/10 18:53:57 $
 *  $Revision: 1.1.2.1 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTROS8Unpacker.h>

#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>

using namespace std;

#include <iostream>

#define SLINK_WORD_SIZE 8


void DTROS8Unpacker::interpretRawData(const unsigned char* index, int datasize,
				      edm::ESHandle<DTReadOutMapping>& mapping, 
				      std::auto_ptr<DTDigiCollection>& product) {


  /// CopyAndPaste from P.Ronchese unpacker

  int wordLength = 4;
  int nrb = datasize / wordLength;
  int rob = 0;
  int ros = 0;
  int mbs[8];
  int lbs[8];
  int err[8];

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
    if ( type == 14 ) {
      int cmap =   word &        0xFF;
      int lock = ( word >> 8 ) & 0xFF;
      int bit;
      for ( bit = 0; bit < 8; bit++ ) {
        mbs[bit] = cmap & 1;
        cmap >>= 1;
        lbs[bit] = lock & 1;
        lock >>= 1; 
        err[bit] = mbs[bit] && lbs[bit];
      }
    }

    // TDC Header/Trailer
    if ( type <= 3 ) {
      int itdc = ( word >> 24 ) & 0xF;
      int ievt = ( word >> 12 ) & 0xFFF;
      int ibwc =   word &         0xFFF;
    }

    // TDC Measurement
    if ( type >= 4 && type <= 5 ) {
      int itdc = ( word >> 24 ) & 0xF;
      int icha = ( word >> 19 ) & 0x1F;
      int time =   word         & 0x7FFFF;

      int edge = ( type == 4 ? 0 : 1 );
      time >>= 2;

      // Map the RO channel to the DetId and wire
      DTDetId layer; int wire = 0; 
      int dduID = 1;
      //mapping->getId(dduID, rosID, robID, tdcID, tdcChannel, layer, wire);
      
      // Produce the digi
      DTDigi digi( time, wire);
      product->insertDigi(layer,digi);
    }
    
  }
  
}
