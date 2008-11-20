/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Logger.h
 *
 *    Description:  Histo Provider to EventProcessor
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:26:04 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Logger_H
#define CSCDQM_Logger_H

#include <iostream>

#ifdef DQMGLOBAL

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#define LOG_DEBUG       ((!edm::MessageDrop::instance()->debugEnabled) ? \
                        LogDebugger() : LogDebugger(true))

#endif

#ifdef DQMLOCAL

#define LOG_DEBUG       LogDebugger()

#endif

#define LOG_ERROR       cscdqm::LogError()
#define LOG_WARN        cscdqm::LogWarn()
#define LOG_INFO        cscdqm::LogInfo()
#define LOG_COUT        cscdqm::LogCout()

namespace cscdqm {

  class Logger { };

//#ifdef DQMGLOBAL

  class LogInfo : public edm::LogInfo, public Logger {
    public: LogInfo() : edm::LogInfo("") { }
  };

  class LogWarn : public edm::LogWarning, public Logger {
    public: LogWarn() : edm::LogWarning("") { }
  };

  class LogError : public edm::LogError, public Logger {
    public: LogError() : edm::LogError("") { }
  };

  class LogDebugger : public edm::LogDebug_, public Logger {
    public: 
      LogDebugger() : edm::LogDebug_() { }
      LogDebugger(const bool& b) : edm::LogDebug_("", __FILE__, __LINE__) { }
  };

//#endif

  class LogCout : public Logger {
    public:

      LogCout() {}

      template< class T >
      LogCout& operator<< (T const & t) { 
        std::cout << t;
        return *this; 
      }

      LogCout& operator<< (std::ostream&(*f)(std::ostream&)) { 
        std::cout << f;
        return *this; 
      }

      LogCout& operator<< (std::ios_base&(*f)(std::ios_base&) ) { 
        std::cout << f;
        return *this; 
      }     

  };

}

#endif
