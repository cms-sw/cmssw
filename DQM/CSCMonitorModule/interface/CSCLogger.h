#ifndef _CSCLogger_h
#define _CSCLogger_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/// === For compatibility with Local DQM
#define LOG4CPLUS_INFO(logger, logEvent) edm::LogInfo (logger) << logEvent;
#define LOG4CPLUS_ERROR(logger, logEvent) edm::LogError (logger) << logEvent;
#define LOG4CPLUS_WARN(logger, logEvent) edm::LogWarning (logger) << logEvent;
#define LOG4CPLUS_DEBUG(logger, logEvent) edm::LogInfo (logger) << logEvent;
#define LOG4CPLUS_FATAL(logger, logEvent) edm::LogFatal (logger) << logEvent;


#endif
