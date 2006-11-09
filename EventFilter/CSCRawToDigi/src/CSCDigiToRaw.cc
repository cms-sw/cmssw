/** \file
 *
 *  $Date: 2006/11/09 19:15:55 $
 *  $Revision: 1.0 $
 *  \author A. Tumanov - Rice
 */

#include <EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>


using namespace edm;
using namespace std;

CSCDigiToRaw::CSCDigiToRaw(){}


CSCDigiToRaw::~CSCDigiToRaw(){}


void CSCDigiToRaw::createFedBuffers(const CSCStripDigiCollection& digis, 
				   FEDRawDataCollection& fed_buffers){
  //...
}
