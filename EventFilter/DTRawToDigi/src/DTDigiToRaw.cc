/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <EventFilter/DTRawToDigi/src/DTDigiToRaw.h>
#include <EventFilter/DTRawToDigi/src/DTDDUWords.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

using namespace edm;
using namespace std;

DTDigiToRaw::DTDigiToRaw(){}


DTDigiToRaw::~DTDigiToRaw(){}


void DTDigiToRaw::createFedBuffers(const DTDigiCollection& digis, 
				   FEDRawDataCollection& fed_buffers){
  //...
}

