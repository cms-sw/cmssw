#include "RecoLuminosity/TCPReceiver/interface/TimeStamp.h"

#include <sstream>
#include <iomanip>

std::string HCAL_HLX::TimeStamp::TimeStampLong( time_t rawtime ){

  if( rawtime == 0 )
    time(&rawtime);

  return ctime(&rawtime);
}

std::string HCAL_HLX::TimeStamp::TimeStampYYYYMM( time_t rawtime ){

  std::string tempStr = TimeStampYYYYMMDD( rawtime );
  return tempStr.substr(0,6);
}

std::string HCAL_HLX::TimeStamp::TimeStampYYYYMMDD( time_t rawtime ){

  if( rawtime == 0 ) 
    time(&rawtime);

  struct tm* timeinfo;
  timeinfo = localtime(&rawtime);

  std::ostringstream out;
  out.str(std::string());
  out << std::setfill('0') << std::setw(4) << timeinfo->tm_year + 1900;
  out << std::setfill('0') << std::setw(2) << timeinfo->tm_mon + 1;
  out << std::setfill('0') << std::setw(2) << timeinfo->tm_mday;

  return out.str();
}
