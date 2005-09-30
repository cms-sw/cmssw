
/* \file DaqFUReader.cc
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "DaqFUReader.h"

#include <FWCore/ParameterSet/interface/ParameterSet.h>

//#include "FURawEvent" 

using namespace std;
using namespace edm;
using namespace raw;


DaqFUReader::DaqFUReader(const ParameterSet& pset){}


DaqFUReader::~DaqFUReader(){}


bool DaqFUReader::fillRawData(EventID& eID,
			      Timestamp& tstamp, 
			      FEDRawDataCollection& data){
  // prendera' FURawEvent* da FilterUnitFramework::rqstEvent()
  return true;
}

