#ifndef Fireworks_Core_fwLog_h
#define Fireworks_Core_fwLog_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     fwLog
//
/**\class fwLog fwLog.h Fireworks/Core/interface/fwLog.h

 Description: Simple logging utility

 Usage:
    To send a message to the logger
       fwLog(kDebug) << "This is my message"<<std::endl;
 
    To change the message levels which will be recorded
       fwlog::setPresentLevel(kDebug);
 
    To change where the messages go, just pass the address of an instance of std::ostream
       fwlog::setLogger(&std::cerr);

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  8 23:10:04 CST 2009
//

// system include files
#include <iostream>

// user include files

// forward declarations
namespace fwlog {
  enum LogLevel { kDebug, kInfo, kWarning, kError };

  const char* levelName(LogLevel);
  std::ostream& logger();
  void setLogger(std::ostream*);

  LogLevel presentLogLevel();
  void setPresentLogLevel(LogLevel);
}  // namespace fwlog

#define fwLog(_level_) \
  (fwlog::presentLogLevel() > _level_) ? fwlog::logger() : fwlog::logger() << fwlog::levelName(_level_) << ": "

#endif
