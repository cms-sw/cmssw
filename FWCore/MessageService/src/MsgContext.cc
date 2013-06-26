// ----------------------------------------------------------------------
//
// MessageLoggerScribe.cc
//
// Changes:
//
// 0 - 6/27/06 mf - created this file to contain compress(), to convert from 
//                  Run: 124 Event: 4567 to 124/4567 for purposes of 
//		    ELstatistics output.


#include "FWCore/MessageService/interface/MsgContext.h"
#include <sstream>

#include <iostream>

namespace edm {
namespace service {       

  std::string MsgContext::compress (const std::string& c) const
  {
    if ( c.substr (0,4) != "Run:" ) return c;
    std::istringstream is (c);
    std::string runWord;
    int run;
    is >> runWord >> run;
    if (!is) return c;
    if (runWord != "Run:") return c;
    std::string eventWord;
    int event;
    is >> eventWord >> event;
    if (!is) return c;
    if (eventWord != "Event:") return c;
    std::ostringstream os;
    os << run << "/" << event;
    return os.str();    
  }
}        // end of namespace service
}       // end of namespace edm
