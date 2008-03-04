#include "EventFilter/CSCRawToDigi/interface/CSCChamberDataItr.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"

CSCChamberDataItr::CSCChamberDataItr(const char * buf) :
  theDCCData(0),
  theCurrentDDU(0)
{
  // first try if it's DCC data.
  const CSCDCCHeader * dccHeader 
    = reinterpret_cast<const CSCDCCHeader *>(buf);
  if(dccHeader->check())
    {
      theDCCData = new CSCDCCEventData((unsigned short *)buf);
      theNumberOfDDUs = theDCCData->dduData().size();
      theDDUItr = new CSCDDUDataItr( &(theDCCData->dduData()[theCurrentDDU]) );
    }
  else 
    {
      // it's DDU data, with only one DDU
      theDDUItr = new CSCDDUDataItr(buf);
      theNumberOfDDUs = 1;
    }
}
  

CSCChamberDataItr::~CSCChamberDataItr() 
{
   // safe, even if it's zero
   delete theDCCData;
}


bool CSCChamberDataItr::next() 
{
  bool result = true;
  if(!theDDUItr->next()) 
    {
      if(++theCurrentDDU >= theNumberOfDDUs)
	{
	  result = false;
	}
      else
	{
	  // the next DDU exists, so initialize an itr
	  assert(theDCCData != 0);
	  delete theDDUItr;
	  theDDUItr = new CSCDDUDataItr( &(theDCCData->dduData()[theCurrentDDU]) );
	}
    }
  return result;
}


const CSCEventData &  CSCChamberDataItr::operator*() 
{
  return **theDDUItr;
}

 
