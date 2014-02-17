//$Id: Utils.cc,v 1.20 2012/04/04 12:17:05 mommsen Exp $
/// @file: Utils.cc

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "boost/date_time/c_local_time_adjustor.hpp"

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
    
    std::string timeStamp(TimePoint_t theTime)
    {
      typedef boost::date_time::c_local_adjustor<boost::posix_time::ptime> local_adj;
      tm ptm = boost::posix_time::to_tm( local_adj::utc_to_local(theTime) );
      std::ostringstream timeStampStr;
      timeStampStr << std::setfill('0') << std::setw(2) << ptm.tm_mday      << "/" 
                   << std::setfill('0') << std::setw(2) << ptm.tm_mon+1     << "/"
                   << std::setfill('0') << std::setw(4) << ptm.tm_year+1900 << ":"
                   << std::setfill('0') << std::setw(2) << ptm.tm_hour      << "/"
                   << std::setfill('0') << std::setw(2) << ptm.tm_min       << "/"
                   << std::setfill('0') << std::setw(2) << ptm.tm_sec;
      return timeStampStr.str();
    }


    std::string timeStampUTC(TimePoint_t theTime)
    {
      tm ptm = to_tm(theTime);
      std::ostringstream timeStampStr;
      timeStampStr << std::setfill('0') << std::setw(2) << ptm.tm_hour      << ":"
                   << std::setfill('0') << std::setw(2) << ptm.tm_min       << ":"
                   << std::setfill('0') << std::setw(2) << ptm.tm_sec;
      return timeStampStr.str();
   }


    std::string asctimeUTC(TimePoint_t theTime)
    {
      tm ptm =  to_tm(theTime);
      char buf[30];
      asctime_r(&ptm, buf);
      std::ostringstream dateStampStr;
      dateStampStr << buf << " UTC";
      return dateStampStr.str();
    }


    std::string dateStamp(TimePoint_t theTime)
    {
      typedef boost::date_time::c_local_adjustor<boost::posix_time::ptime> local_adj;
      tm ptm = boost::posix_time::to_tm( local_adj::utc_to_local(theTime) );
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
      #if linux
      struct stat64 results;
      int retVal = stat64(path.c_str(), &results);
      #else
      struct stat results;
      int retVal = stat(path.c_str(), &results);
      #endif

      if( retVal != 0 )
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
