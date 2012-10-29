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
#include <iomanip>
//#include <typeinfo> typeid(this).name()

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#ifdef DQMGLOBAL

#define LOG_DEBUG       ((!edm::MessageDrop::instance()->debugEnabled) ? \
                        cscdqm::LogDebugger(false) : cscdqm::LogDebugger())

#endif

#ifdef DQMLOCAL

#define LOG_DEBUG       cscdqm::LogDebugger()

#endif

#define LOG_ERROR       cscdqm::LogError()
#define LOG_WARN        cscdqm::LogWarn()
#define LOG_INFO        cscdqm::LogInfo()
#define LOG_COUT        cscdqm::LogCout()

namespace cscdqm {

  /**
   * @class Logger
   * @brief Base Logger Object (empty)
   */
  class Logger { };

  /**
   * @class LogInfo
   * @brief Information level logger. Use LOG_INFO macros instead, i.e.
   * LOG_INFO << "x = " << x;
   */
  class LogInfo : public edm::LogInfo, public Logger {
#ifdef DQMGLOBAL
    public: LogInfo() : edm::LogInfo("") { }
#else
    public: LogInfo() : edm::LogInfo() { }
#endif
  };

  /**
   * @class LogWarn
   * @brief Warning level logger. Use LOG_WARN macros instead, i.e. LOG_WARN
   * << "x = " << x;
   */
  class LogWarn : public edm::LogWarning, public Logger {
#ifdef DQMGLOBAL
    public: LogWarn() : edm::LogWarning("") { }
#else
    public: LogWarn() : edm::LogWarning() { }
#endif
  };

  /**
   * @class LogError
   * @brief Error level logger. Use LOG_ERROR macros instead, i.e. LOG_ERROR <<
   * "x = " << x;
   */
  class LogError : public edm::LogError, public Logger {
#ifdef DQMGLOBAL
    public: LogError() : edm::LogError("") { }
#else
    public: LogError() : edm::LogError() { }
#endif
  };

#ifdef DQMGLOBAL

  /**
   * @class LogDebugger
   * @brief Debug Level logger. Use LOG_DEBUG macros instead, i.e. LOG_DEBUG <<
   * "x = " << x;
   */
  class LogDebugger : public edm::LogDebug_, public Logger {
    public: 
      LogDebugger() : edm::LogDebug_("", __FILE__, __LINE__) { }
      LogDebugger(const bool empty) : edm::LogDebug_() { }
  };

#endif

#ifdef DQMLOCAL

  /**
   * @class LogDebugger
   * @brief Debug Level logger. Use LOG_DEBUG macros instead, i.e. LOG_DEBUG <<
   * "x = " << x;
   */
  class LogDebugger : public edm::LogDebug, public Logger {
    public: 
      LogDebugger() : edm::LogDebug() { }
  };

#endif

  /**
   * @class LogCout
   * @brief Simple logger that prints stuff to std::cout. Use LOG_COUT macros
   * instead, i.e. LOG_COUT << "x = " << x;
   */
  class LogCout : public Logger {
    public:

      LogCout() { }
      ~LogCout() { std::cout << std::endl; }

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
