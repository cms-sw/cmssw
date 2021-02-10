#ifndef DIPTIMESTAMP_H_INCLUDED
#define DIPTIMESTAMP_H_INCLUDED
#include "Options.h"
#include "StdTypes.h"
#include <time.h>
#include <stdio.h>
#include <sys/timeb.h>

/**
* used to hold data information that indicates when published DIP data wasd obtained.
*/
class DipDllExp DipTimestamp 
{
	superlong timeNanoseconds; // UTC nanoseconds

public:

	/**
	* Creats time stamp object with the current time.
	* */
	DipTimestamp(){
		setCurrentTime();
	}
	


  /**
   * Creates DipTimestamp from UTC Java time (in milliseconds).
   */
	DipTimestamp(superlong timestamp_ms) 
	{
		timeNanoseconds = timestamp_ms*1000000;
	}


	/**
	* Set time object to the current time
	*/
	void setCurrentTime(){
		struct timeb timebuf;
		superlong secs;
		superlong millis;

		ftime(&timebuf);
		secs = timebuf.time;
		millis = (secs * 1000)+timebuf.millitm;
		timeNanoseconds = millis*1000000;
	}


	/**
	* The time object to the time whos value is passed in Nano Seconds
	*/
	void setMillis(superlong miliseconds) 
	{
		timeNanoseconds = miliseconds*1000000;
	}



	/**
	* The time object to the time whos value is passed in Nano Seconds
	*/
	void setNanos(superlong nanoseconds) 
	{
		timeNanoseconds = nanoseconds;
	}


	/**
	* get the time held by the object in UTC mSec
	*/
	superlong getAsMillis() const
	{
		return timeNanoseconds/1000000;
	}


	/**
	* get the time held by the object in UTC nanoSec
	*/
	superlong getAsNanos() const
	{
		return timeNanoseconds;
	}
};

#endif //DIPTIMESTAMP_H_INCLUDED
