#include "EventFilter/Utilities/interface/RateStat.h"
#include "EventFilter/Utilities/interface/CurlPoster.h"
#include "EventFilter/Utilities/interface/DebugUtils.h"

#include <iostream>

namespace evf{

  RateStat::RateStat(std::string iDieUrl) : iDieUrl_(iDieUrl)
  {
  poster_ = new CurlPoster(iDieUrl_);
  }
  
  RateStat::~RateStat()
  {
    delete poster_;
  }

    void RateStat::sendStat(const unsigned char *buf, size_t len, unsigned int lsid)
  {
    poster_->postBinary(buf,len,lsid);
  }
  
  void RateStat::sendLegenda(const std::string &message)
  {
    poster_->postString(message.c_str(),message.length(),0,CurlPoster::leg);
  }

}
