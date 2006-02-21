/** \file
 *
 *  $Date: 2005/10/26 11:35:25 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <EventFilter/DTRawToDigi/src/DTDigiToRaw.h>
#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
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

