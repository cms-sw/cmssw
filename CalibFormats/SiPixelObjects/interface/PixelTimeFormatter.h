#ifndef PixelTimeFormatter_h
#define PixelTimeFormatter_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h
* \brief This class provides utility methods to manipulate ASCII formatted timestamps
*
*   A longer explanation will be placed here later
*/

#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <sys/time.h>

namespace pos{
  class PixelTimeFormatter
  {
  public:
    static std::string getTime(void) 
    {
      char theDate[20] ;
      struct tm *thisTime;
      time_t aclock;
      std::string date ;
      time( &aclock );		  
      thisTime = localtime( &aclock ); 
       
      sprintf(theDate,
	      "%d-%02d-%02d %02d:%02d:%02d", thisTime->tm_year+1900,
	      thisTime->tm_mon+1,
	      thisTime->tm_mday,
	      thisTime->tm_hour,
	      thisTime->tm_min,
	      thisTime->tm_sec ); 
      date = theDate ;
      std::cout << "[PixelTimeFormatter::getTime()]\t\t\t\t    Time: " << date << std::endl ;					  
      return date ;
    }

    static std::string getmSecTime(void) 
    {
      char theDate[20] ;
      struct timeval msecTime;
      gettimeofday(&msecTime, (struct timezone *)0) ;
      
      sprintf(theDate,
	      "%d-%d", 
	      (unsigned int)msecTime.tv_sec,
	      (unsigned int)msecTime.tv_usec ); 
      return std::string(theDate) ;
    }
  } ;
}

#endif
