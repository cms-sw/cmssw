/*************************************************************************
 * XDAQ Components for Pixel Online Software                             *
 * Authors: Dario Menasce, Marco Rovere                                  *
 ************************************************************************/

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
#include <cstdlib>

#define USE_TIMER_ 0

namespace pos{
  class PixelTimeFormatter
  {
  public:

    //---------------------------------------------------------------------------------
    PixelTimeFormatter(std::string source ) 
    {
     if( !USE_TIMER_) return ;
     origin_ = source ;
     std::cout << "[PixelTimeFormatter::PixelTimeFormatter()]\t\t    Time counter started for " << origin_ << std::endl ;
     startTime_ = getImSecTime() ;
    }
    
    void stopTimer(void) 
    {
     if( !USE_TIMER_ ) return ;
     endTime_ = getImSecTime() ;
     double start = startTime_.tv_sec + startTime_.tv_usec/1000000. ;  
     double stop  = endTime_.tv_sec   + endTime_.tv_usec/1000000. ;  
     std::cout << "[PixelTimeFormatter::stopTimer()]\t\t\t    Elapsed time: " << stop-start << " seconds for " << origin_ << std::endl ;
    }
    
    virtual void writeXMLHeader(pos::PixelConfigKey key,
                                int version, std::string path,
                                std::ofstream *out,
                                std::ofstream *out1 = NULL,
                                std::ofstream *out2 = NULL
                                ) const {;}

    //---------------------------------------------------------------------------------
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
      //std::cout << "[PixelTimeFormatter::getTime()]\t\t\t\t    Time: " << date << std::endl ;					  
      return date ;
    }

    //---------------------------------------------------------------------------------
    struct tm * getITime(void) 
    {
      struct tm *thisTime;
      time_t aclock;
      time( &aclock );		  
      thisTime = localtime( &aclock ); 
      return thisTime ;
    }

    //---------------------------------------------------------------------------------
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

    //---------------------------------------------------------------------------------
    struct timeval getImSecTime(void) 
    {
      struct timeval msecTime;
      gettimeofday(&msecTime, (struct timezone *)0) ;
      
      return msecTime ;
    }
/*    
    //---------------------------------------------------------------------------------
    static double timeDiff(std::string firstTime, std::string secondTime)
    {
      time_t rawTime;
      struct tm * rawTimeInfo;

      int firstMonth  = atoi( firstTime.substr(0,2).c_str()) ;
      int firstDay    = atoi( firstTime.substr(3,2).c_str()) ;
      int firstYear   = atoi( firstTime.substr(6,4).c_str()) ;
      int secondMonth = atoi(secondTime.substr(0,2).c_str()) ;
      int secondDay   = atoi(secondTime.substr(3,2).c_str()) ;
      int secondYear  = atoi(secondTime.substr(6,4).c_str()) ;
  
      time(&rawTime);
      rawTimeInfo = localtime(&rawTime);
      rawTimeInfo->tm_mon  = firstMonth - 1    ;
      rawTimeInfo->tm_mday = firstDay	       ;
      rawTimeInfo->tm_year = firstYear  - 1900 ;

      time_t ft = mktime( rawTimeInfo ) ;

      rawTimeInfo = localtime(&rawTime);
      rawTimeInfo->tm_mon  = secondMonth - 1	;
      rawTimeInfo->tm_mday = secondDay  	;
      rawTimeInfo->tm_year = secondYear  - 1900 ;

      time_t st = mktime( rawTimeInfo ) ;
  
      return difftime(ft, st) ;
    }
*/    
    //=================================================================================
    
    private:
    
     struct timeval startTime_      ;
     struct timeval endTime_        ;
     std::string    origin_         ;
     bool           verbose_        ;
  } ;
}

#endif
