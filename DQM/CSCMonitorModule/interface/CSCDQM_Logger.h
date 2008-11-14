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
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#define LOG_ERROR       LogError()
#define LOG_WARN        LogWarn()
#define LOG_INFO        LogInfo()
#define LOG_DEBUG       ((!edm::MessageDrop::instance()->debugEnabled) ? \
                        LogDebugger() : LogDebugger(true))
#define LOG_COUT        LogCout()

namespace cscdqm {

  class LogInfo : public edm::LogInfo {
    public: LogInfo() : edm::LogInfo("") { }
  };

  class LogWarn : public edm::LogWarning {
    public: LogWarn() : edm::LogWarning("") { }
  };

  class LogError : public edm::LogError {
    public: LogError() : edm::LogError("") { }
  };

  class LogDebugger : public edm::LogDebug_ {
    public: 
      LogDebugger() : edm::LogDebug_() { }
      LogDebugger(const bool& b) : edm::LogDebug_("", __FILE__, __LINE__) { }
  };

  class LogCout {
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
