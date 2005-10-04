
/* \file DaqFUReader.cc
 *
 *  $Date: 2005/09/30 08:17:48 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include "DaqFUReader.h"

#include <FWCore/ParameterSet/interface/ParameterSet.h>

//#include "FURawEvent" 

using namespace std;
using namespace edm;


DaqFUReader::DaqFUReader(const ParameterSet& pset){}


DaqFUReader::~DaqFUReader(){}


bool DaqFUReader::fillRawData(EventID& eID,
			      Timestamp& tstamp, 
			      FEDRawDataCollection& data){
  // prendera' FURawEvent* da FilterUnitFramework::rqstEvent()
  return true;
}

