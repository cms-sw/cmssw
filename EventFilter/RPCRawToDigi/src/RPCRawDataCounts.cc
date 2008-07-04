#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"

#include <vector>
#include <iostream>

using namespace std;

void RPCRawDataCounts::addRecordType(int fed, int type)
{
  if (theRecordTypes.find(fed) == theRecordTypes.end()) {
    theRecordTypes[fed]=vector<int>( 10,0);
  }
  vector<int> & v = theRecordTypes[fed]; 
  v[type] +=1;
}

void RPCRawDataCounts::addReadoutError(int error)
{
  if ( theReadoutErrors.find(error) == theReadoutErrors.end() ) theReadoutErrors[error]=0;
  theReadoutErrors[error] += 1;
}
