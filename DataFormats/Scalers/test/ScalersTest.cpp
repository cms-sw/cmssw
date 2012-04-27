//
// File: ScalersTest.cpp         (W.Badgett)
//
// Program to test Scalers RawToDigi conversion from binary raw 
// data file
//

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"


#include <iostream>
#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <cstring>

const char * fileName = "scalers.dat";

int main(int argc, char** argv)
{
  unsigned char buffer [sizeof(struct ScalersEventRecordRaw_v1)];
  int ctr = 0;
  int retcod;
  int bytes = 1;
  const L1TriggerScalers *previousTrig = NULL;
  int fd = open(fileName, O_RDONLY);

  if ( fd > 0 )
  {
    while ( bytes > 0 )
    {
      bytes = read(fd,buffer,sizeof(struct ScalersEventRecordRaw_v1));
      if ( bytes <= 0 )
      {
	retcod = errno;
	if ( retcod == 0 )
	{
	  printf("Finished reading file %s with %d events\n", fileName,
		 ctr);
	}
	else
	{
	  printf("Error %s reading file %s\n", fileName,
		 strerror(errno));
	}
      }
      else
      {
	ctr++;
	printf("\n******* Event %d Read %d bytes from %s *******\n", 
	       ctr, bytes, fileName);
	const L1TriggerScalers *trig = new L1TriggerScalers(buffer);
	std::cout << *trig;

	if ( ctr > 1 )
	{
	  const L1TriggerScalers *previousTrigSave = previousTrig;
	  if ( previousTrig->orbitNumber() <
	       trig->orbitNumber() )
	  {
	    L1TriggerRates rates(*previousTrig,*trig);
	    std::cout << std::endl;
	    std::cout << rates;
	    previousTrig = trig;
	  }
	  delete(previousTrigSave);
	}
	else
	{
	  previousTrig = trig;
	}
	std::cout << std::endl;
	LumiScalers lumi(buffer);
	std::cout << lumi;
      }
    }
    close(fd);
  }
  else
  {
    printf("Error %s opening file %s\n", fileName,
	   strerror(errno));
  }
  return 0;
}
