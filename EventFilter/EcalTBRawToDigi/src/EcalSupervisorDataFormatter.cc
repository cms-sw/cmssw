#include "EcalSupervisorDataFormatter.h"

using namespace edm;
using namespace std;

#include <iostream>

EcalSupervisorDataFormatter::EcalSupervisorDataFormatter () {
}

void EcalSupervisorDataFormatter::interpretRawData( const FEDRawData & fedData, 
					   EcalTBEventHeader& tbEventHeader)
{
  const ulong * buffer = ( reinterpret_cast<ulong*>(const_cast<unsigned char*> ( fedData.data())));
  int fedLenght                        = fedData.size(); // in Bytes
  
  // check ultimate fed size and strip off fed-header and -trailer
   if (fedLenght < (nWordsPerEvent *4) )
     {
       LogError("EcalSupervisorDataFormatter") << "EcalSupervisorData has size "  <<  fedLenght
 				       <<" Bytes as opposed to expected " 
 				       << (nWordsPerEvent *4)
 				       << ". Returning."<< endl;
       return;
     }

  ulong a=1; // used to extract an 8 Bytes word from fed 
  ulong b=1; // used to manipulate the 8 Bytes word and get what needed

  int wordCounter =0;
  a = buffer[wordCounter];wordCounter++;
  b = (a & 0xfff00000);
  b = b >> 20;
  tbEventHeader.setBurstNumber(b);
  LogDebug("EcalSupervisorDataFormatter") << "Burst number:\t" << b << endl;
  //Skipping the second word
  wordCounter +=1;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0x80000000);
  b = b >> 31;
  tbEventHeader.setSyncError(b & 0x1);
  LogDebug("EcalSupervisorDataFormatter") << "Sync Error:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffff);
  tbEventHeader.setRunNumber(b);
  LogDebug("EcalSupervisorDataFormatter") << "Run Number:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xff);
  int version = b;
  LogDebug("EcalSupervisorDataFormatter") << "Version Number:\t" << b << endl;

  int numberOfMagnetMeasurements = -1;
  if (version >= 11)
    {
      b = (a& 0xff00);
      b = b >> 8;
      numberOfMagnetMeasurements= b;
      tbEventHeader.setNumberOfMagnetMeasurements(b);
      LogDebug("EcalSupervisorDataFormatter") << "Number Of Magnet Measurements:\t" << b << endl;
    }

  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEventNumber(b);
  LogDebug("EcalSupervisorDataFormatter") << "Event Number:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setBegBurstTimeSec(b);
  LogDebug("EcalSupervisorDataFormatter") << "BegBurstTimeSec:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setBegBurstTimeMsec(b);
  LogDebug("EcalSupervisorDataFormatter") << "BegBurstTimeMsec:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEndBurstTimeSec(b);
  LogDebug("EcalSupervisorDataFormatter") << "EndBurstTimeSec:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEndBurstTimeMsec(b);
  LogDebug("EcalSupervisorDataFormatter") << "EndBurstTimeMsec:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setBegBurstLV1A(b);
  LogDebug("EcalSupervisorDataFormatter") << "BegBurstLV1A:\t" << b << endl;
  a = buffer[wordCounter];wordCounter++;
  b = (a& 0xffffffff);
  tbEventHeader.setEndBurstLV1A(b);
  LogDebug("EcalSupervisorDataFormatter") << "EndBurstLV1A:\t" << b << endl;

  if (version >= 11)
    {
      std::vector<EcalTBEventHeader::magnetsMeasurement_t> magnetMeasurements;
      for (int iMagMeas = 0; iMagMeas < numberOfMagnetMeasurements; iMagMeas ++)
	{ 
	  LogDebug("EcalSupervisorDataFormatter") << "++++++ New Magnet Measurement++++++\t" << iMagMeas + 1 << endl;
	  EcalTBEventHeader::magnetsMeasurement_t aMeasurement;
	  wordCounter+=4;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6IRead_ampere = b;
	  LogDebug("EcalSupervisorDataFormatter") << "NominalMagnet6ReadAmpere:\t" << b << endl;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6ISet_ampere = b;
	  LogDebug("EcalSupervisorDataFormatter") << "NominalMagnet6SetAmpere:\t" << b << endl;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7IRead_ampere = b;
	  LogDebug("EcalSupervisorDataFormatter") << "NominalMagnet7ReadAmpere:\t" << b << endl;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7ISet_ampere = b;
	  LogDebug("EcalSupervisorDataFormatter") << "NominalMagnet7SetAmpere:\t" << b << endl;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7VMeas_uvolt = b;
	  LogDebug("EcalSupervisorDataFormatter") << "MeasuredMagnet7MicroVolt:\t" << b << endl;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet7IMeas_uampere = b;
	  LogDebug("EcalSupervisorDataFormatter") << "MeasuredMagnet7Ampere:\t" << b << endl;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6VMeas_uvolt = b;
	  LogDebug("EcalSupervisorDataFormatter") << "MeasuredMagnet6MicroVolt:\t" << b << endl;
	  a = buffer[wordCounter];wordCounter++;
	  b = (a& 0xffffffff);
	  aMeasurement.magnet6IMeas_uampere = b;
	  LogDebug("EcalSupervisorDataFormatter") << "MeasuredMagnet6Ampere:\t" << b << endl;
	  magnetMeasurements.push_back(aMeasurement);
	}
      tbEventHeader.setMagnetMeasurements(magnetMeasurements);
    }
}
