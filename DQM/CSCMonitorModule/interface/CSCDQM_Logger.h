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

namespace cscdqm {

  typedef enum LogType { 
    ERROR, 
    WARNING, 
    INFO, 
    DEBUG 
  };

class Logger {
  public:

    explicit Logger(LogType type){ this->type = type; }

    template< class T >
    Logger& operator<< (T const & t) { 
      switch(type) {
        case ERROR:
          edm::LogError("Error") << t;
          break;
        case WARNING:
          edm::LogWarning("Warning") << t;
          break;
        case INFO:
          edm::LogInfo("Info") << t;
          break;
        case DEBUG:
          LogDebug("Debug") << t;
          break;
      }
      return *this; 
    }

    Logger& operator<< (std::ostream&(*f)(std::ostream&)) { 
      switch(type) {
        case ERROR:
          edm::LogError ("Error") << f;
          break;
        case WARNING:
          edm::LogWarning ("Warning") << f;
          break;
        case INFO:
          edm::LogInfo ("Info") << f;
          break;
        case DEBUG:
          LogDebug ("Debug") << f;
          break;
      }
      return *this; 
    }

    Logger& operator<< (std::ios_base&(*f)(std::ios_base&) ) { 
      switch(type) {
        case ERROR:
          edm::LogError ("Error") << f;
          break;
        case WARNING:
          edm::LogWarning ("Warning") << f;
          break;
        case INFO:
          edm::LogInfo ("Info") << f;
          break;
        case DEBUG:
          LogDebug ("Debug") << f;
          break;
      }
      return *this; 
    }     

  private:

    LogType type;
};

}

#endif
