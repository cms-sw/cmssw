/** \file
 *
 *  $Date: 2005/11/10 18:53:57 $
 *  $Revision: 1.1.2.1 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTDDUUnpacker.h>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>

using namespace std;

#include <iostream>

#define SLINK_WORD_SIZE 8


void DTDDUUnpacker::interpretRawData(const unsigned char* index, int datasize) {

  // Check DDU header
  FEDHeader dduHeader(index);

  // Check DDU trailer
  FEDTrailer dduTrailer(index + datasize - SLINK_WORD_SIZE);

  // Check Status Words
  DTDDUFirstStatusWord dduStatusWord1(index + datasize - 3*SLINK_WORD_SIZE);
  
  DTDDUSecondStatusWord dduStatusWord2(index + datasize - 2*SLINK_WORD_SIZE);

}
