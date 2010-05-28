//$Id: Utils.cc,v 1.13 2010/04/13 09:15:39 mommsen Exp $
/// @file: Utils.cc

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include <iomanip>
#include <sstream>
#include <string.h>

#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>


namespace stor
{
  namespace utils
  {
    
    namespace
    {
      /**
	 Convert a (POSIX) timeval into a time_point_t.
      */
      inline void timeval_to_timepoint(timeval const& in, 
				       time_point_t& out)
      {
	// First set the seconds.
	out = static_cast<time_point_t>(in.tv_sec);
	
	// Then set the microseconds.
	out += static_cast<time_point_t>(in.tv_usec)/(1000*1000);
      }
    } // anonymous namespace
      
      
    time_point_t getCurrentTime()
    {
      time_point_t result = -1.0;
      timeval now;
      if (gettimeofday(&now, 0) == 0) timeval_to_timepoint(now, result);
      return result;
    }

    int sleep(duration_t interval)
    {
      if (interval < 0) return -1;
      timespec rqtp;
      rqtp.tv_sec = static_cast<time_t>(interval); // truncate
      rqtp.tv_nsec = static_cast<long>((interval-rqtp.tv_sec)*1000*1000*1000);
      return nanosleep(&rqtp, 0);
    }

    int sleepUntil(time_point_t theTime)
    {
      time_point_t now = getCurrentTime();
      duration_t interval = theTime - now;
      return sleep(interval);
    }
    
    std::string timeStamp(time_point_t theTime)
    {
      time_t rawtime = (time_t)theTime;
      tm ptm;
      localtime_r(&rawtime, &ptm);
      std::ostringstream timeStampStr;
      std::string colon(":");
      std::string slash("/");
      timeStampStr << std::setfill('0') << std::setw(2) << ptm.tm_mday      << slash 
                   << std::setfill('0') << std::setw(2) << ptm.tm_mon+1     << slash
                   << std::setfill('0') << std::setw(4) << ptm.tm_year+1900 << colon
                   << std::setfill('0') << std::setw(2) << ptm.tm_hour      << slash
                   << std::setfill('0') << std::setw(2) << ptm.tm_min       << slash
                   << std::setfill('0') << std::setw(2) << ptm.tm_sec;
      return timeStampStr.str();
    }


    std::string dateStamp(time_point_t theTime)
    {
      time_t rawtime = (time_t)theTime;
      tm ptm;
      localtime_r(&rawtime, &ptm);
      std::ostringstream dateStampStr;
      dateStampStr << std::setfill('0') << std::setw(4) << ptm.tm_year+1900
                   << std::setfill('0') << std::setw(2) << ptm.tm_mon+1
                   << std::setfill('0') << std::setw(2) << ptm.tm_mday;
      return dateStampStr.str();
    }

    
    std::string getIdentifier(xdaq::ApplicationDescriptor *appDesc)
    {
      std::ostringstream identifier;
      identifier << appDesc->getClassName() << appDesc->getInstance() << "/";
      return identifier.str();
    }
    
    
    void checkDirectory(const std::string& path)
    {
      struct stat64 results;
      
      int retVal = stat64(path.c_str(), &results);
      if( retVal !=0 )
      {
        std::ostringstream msg;
        msg << "Directory " << path << " does not exist: " << strerror(errno);
        XCEPT_RAISE(stor::exception::NoSuchDirectory, msg.str());
      }
      if ( !(results.st_mode & S_IWUSR) )
      {
        std::ostringstream msg;
        msg << "Directory " << path << " is not writable.";
        XCEPT_RAISE(stor::exception::NoSuchDirectory, msg.str());
      }
    }


    void getStdVector(xdata::Vector<xdata::String>& x, std::vector<std::string>& s)
    {
      s.clear();
      s.reserve(x.elements());
      for(xdata::Vector<xdata::String>::iterator it = x.begin(),
            itEnd = x.end();
          it != itEnd;
          ++it)
      {
        s.push_back( it->toString() );
      }
    }


    void getXdataVector(const std::vector<std::string>& v, xdata::Vector<xdata::String>& x)
    {
      x.clear();
      x.reserve(v.size());
      for(std::vector<std::string>::const_iterator it = v.begin(),
            itEnd = v.end();
          it != itEnd;
          ++it)
      {
        x.push_back( static_cast<xdata::String>(*it) );
      }
    }


  } // namespace utils

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
