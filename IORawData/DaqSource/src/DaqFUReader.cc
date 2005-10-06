/** \file
 *
 *  $Date: 2005/10/04 18:38:48 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include "DaqFUReader.h"

#include <FWCore/ParameterSet/interface/ParameterSet.h>


using namespace std;
using namespace edm;


DaqFUReader::DaqFUReader(const ParameterSet& pset){}


DaqFUReader::~DaqFUReader(){}


bool DaqFUReader::fillRawData(EventID& eID,
			      Timestamp& tstamp, 
			      FEDRawDataCollection& data){
  //cf. FURawEvent* da FilterUnitFramework::rqstEvent()
  return true;
}

