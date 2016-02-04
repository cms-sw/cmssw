#include "EcalSupervisorDataFormatter.h"


#include <iostream>

EcalSupervisorTBDataFormatter::EcalSupervisorTBDataFormatter () {
}

void EcalSupervisorTBDataFormatter::interpretRawData( const FEDRawData & fedData, 
					   EcalTBEventHeader& tbEventHeader)
{
  const unsigned long * buffer = ( reinterpret_cast<unsigned long*>(const_cast<unsigned char*> ( fedData.data())));
  int fedLenght                        = fedData.size(); // in Bytes
  
  // check ultimate fed size and strip off fed-header and -trailer
   if (fedLenght < (nWordsPerEvent *4) )
     {
       edm::LogError("EcalSupervisorTBDataFormatter") << "EcalSupervisorTBData has size "  <<  fedLenght
 				       <<" Bytes as opposed to expected " 
 				       << (nWordsPerEvent *4)
 				       << ". Returning.";
       return;
     }

  unsigned long a=1; // used to extract an 8 Bytes word from fed 
  unsigned long b=1; // used to manipulate the 8 Bytes word and get what needed

  int wordCounter =0;
  a = buffer[wordCounter];wordCounter++;
  b = (a & 0xfff00000);
  b = b >> 20;
  tbEventHeader.setBurstNumber(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "Burst number:\t" << b;
  //Skipping the second word
  wordCounter +=1;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0x80000000);
  b = b >> 31;
  tbEventHeader.setSyncError(b & 0x1);
  LogDebug("EcalSupervisorTBDataFormatter") << "Sync Error:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffff);
  tbEventHeader.setRunNumber(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "Run Number:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xff);
  int version = b;
  LogDebug("EcalSupervisorTBDataFormatter") << "Version Number:\t" << b;

  int numberOfMagnetMeasurements = -1;
  if (version >= 11)
    {
      b = (a& 0xff00);
      b = b >> 8;
      numberOfMagnetMeasurements= b;
      tbEventHeader.setNumberOfMagnetMeasurements(b);
      LogDebug("EcalSupervisorTBDataFormatter") << "Number Of Magnet Measurements:\t" << b;
    }

  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEventNumber(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "Event Number:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setBegBurstTimeSec(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "BegBurstTimeSec:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setBegBurstTimeMsec(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "BegBurstTimeMsec:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEndBurstTimeSec(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "EndBurstTimeSec:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEndBurstTimeMsec(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "EndBurstTimeMsec:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setBegBurstLV1A(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "BegBurstLV1A:\t" << b;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEndBurstLV1A(b);
  LogDebug("EcalSupervisorTBDataFormatter") << "EndBurstLV1A:\t" << b;

  if (version >= 11)
    {
      std::vector<EcalTBEventHeader::magnetsMeasurement_t> magnetMeasurements;
      for (int iMagMeas = 0; iMagMeas < numberOfMagnetMeasurements; iMagMeas ++)
	{ 
	  LogDebug("EcalSupervisorTBDataFormatter") << "++++++ New Magnet Measurement++++++\t" << (iMagMeas + 1);
	  EcalTBEventHeader::magnetsMeasurement_t aMeasurement;
	  wordCounter+=4;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6IRead_ampere = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "NominalMagnet6ReadAmpere:\t" << b;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6ISet_ampere = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "NominalMagnet6SetAmpere:\t" << b;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7IRead_ampere = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "NominalMagnet7ReadAmpere:\t" << b;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7ISet_ampere = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "NominalMagnet7SetAmpere:\t" << b;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7VMeas_uvolt = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "MeasuredMagnet7MicroVolt:\t" << b;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7IMeas_uampere = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "MeasuredMagnet7Ampere:\t" << b;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6VMeas_uvolt = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "MeasuredMagnet6MicroVolt:\t" << b;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6IMeas_uampere = b;
	  LogDebug("EcalSupervisorTBDataFormatter") << "MeasuredMagnet6Ampere:\t" << b;
	  magnetMeasurements.push_back(aMeasurement);
	}
      tbEventHeader.setMagnetMeasurements(magnetMeasurements);
    }
}
