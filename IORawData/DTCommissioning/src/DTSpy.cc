/*
 *  DTSpy.cc
 *  
 *
 *  Created by Sandro Ventura on 7/28/07.
 *  Copyright 2007 __MyCompanyName__. All rights reserved.
 *
 */

#include "DTSpy.h"
#include <stdlib.h>
#include <string.h>

#define DTSPY_HEAD_SIZE 36
#define DTSPY_MAX_MSG  512*1024


DTSpy::DTSpy():DTCtcp(0)
{
  spybuf=(char *)malloc(DTSPY_MAX_MSG);
  givenonce=0;
}


DTSpy::~DTSpy()
{
  free(spybuf);
}


DTSpy::DTSpy(char *hostaddr,int port):DTCtcp(0)
{
     Connect(hostaddr,port);
     givenonce=0;
}


int
DTSpy::getNextBuffer()
{

	if (!connected) return -1;
	if (givenonce)
		if ((lastpointer - spybuf) < getBuffSize())
			return (getBuffSize() - (lastpointer - spybuf));
	givenonce=0;

	memset(spybuf,0x0,DTSPY_MAX_MSG); /* init buffer */
    
    	int howm=Receive(spybuf,DTSPY_HEAD_SIZE);
	
	unsigned short *i2ohea = (unsigned short *) (spybuf+2);
	if (howm == DTSPY_HEAD_SIZE ) 
         {
		howm = Receive(spybuf+DTSPY_HEAD_SIZE,((*i2ohea) * 4)-DTSPY_HEAD_SIZE);

               // for (int ii=0; ii<12; ii++)
               //       printf("%d: %x %x %x %x \n",ii*4,spybuf[ii*4],
               //             spybuf[ii*4+1],spybuf[ii*4+2],spybuf[ii*4+3]);

	        return howm+DTSPY_HEAD_SIZE;;
         }
         else return -1;
}

int
DTSpy::getBuffSize()
{
  	unsigned short *i2ohea = (unsigned short *) (spybuf+2);
	return *i2ohea;
}



int
DTSpy::getRunNo()
{
    unsigned short *i2ohea = (unsigned short *) (spybuf+28);
    return *i2ohea;
}

const char *
DTSpy::getEventPointer()
{
  if (givenonce) return lastpointer;
  lastpointer = spybuf+DTSPY_HEAD_SIZE;
  givenonce = 1;
  return lastpointer;
}

void 
DTSpy::setlastPointer(char * thep)
{
  lastpointer=thep;
}
