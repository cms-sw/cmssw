#include "TableDataFormatter.h"


#include <iostream>

TableDataFormatter::TableDataFormatter () {
}

void TableDataFormatter::interpretRawData( const FEDRawData & fedData, 
					   EcalTBEventHeader& tbEventHeader)
{
  const unsigned long * buffer = ( reinterpret_cast<unsigned long*>(const_cast<unsigned char*> ( fedData.data())));
  int fedLenght                        = fedData.size(); // in Bytes
  
  // check ultimate fed size and strip off fed-header and -trailer
  if (fedLenght != (nWordsPerEvent *4) )
    {
      edm::LogError("TableDataFormatter") << "TableData has size "  <<  fedLenght
				       <<" Bytes as opposed to expected " 
				       << (nWordsPerEvent *4)
				     << ". Returning.";
      return;
    }

  unsigned long a=1; // used to extract an 8 Bytes word from fed 
  unsigned long b=1; // used to manipulate the 8 Bytes word and get what needed

  int wordCounter =0;
  wordCounter +=4;

  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setThetaTableIndex(b);
  LogDebug("TableDataFormatter") << "Table theta position:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setPhiTableIndex(b);
  LogDebug("TableDataFormatter") << "Table phi position:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffff);
  tbEventHeader.setCrystalInBeam(EBDetId(1,b,EBDetId::SMCRYSTALMODE));
  LogDebug("TableDataFormatter") << "Actual Current crystal in beam:\t" << b;
  b = (a& 0xffff0000);
  b = b >> 16;
  tbEventHeader.setNominalCrystalInBeam(EBDetId(1,b,EBDetId::SMCRYSTALMODE));
  LogDebug("TableDataFormatter") << "Nominal Current crystal in beam:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffff);
  tbEventHeader.setNextCrystalInBeam(EBDetId(1,b,EBDetId::SMCRYSTALMODE));
  LogDebug("TableDataFormatter") << "Next crystal in beam:\t" << b;
  b = (a& 0x00010000); //Table is moving at the begin of the spill
  b = b >> 16;
  tbEventHeader.setTableIsMovingAtBegSpill(b & 0x1);
  LogDebug("TableDataFormatter") << "Table is moving at begin of the spill:\t" << b;
}
