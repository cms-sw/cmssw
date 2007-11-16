//
// File: ScalersTest.cpp
//

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"


#include <iostream>
#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

char * fileName = "scalers.dat";

int main(int argc, char** argv)
{
  struct ScalersEventRecordRaw_v1 record;
  unsigned char buffer [1024];

  int fd = open(fileName, O_RDONLY);
  if ( fd > 0 )
  {
    int bytes = read(fd,buffer,sizeof(struct ScalersEventRecordRaw_v1));
    if ( bytes <= 0 )
    {
      printf("Error %s reading file %s\n", fileName,
	     strerror(errno));
    }
    else
    {
      printf("Read %d bytes from %s\n", bytes, fileName);
      L1TriggerScalers trig(buffer);
      LumiScalers lumi(buffer);
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
