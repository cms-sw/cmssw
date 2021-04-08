#ifndef CondCore_CondDB_Logger_h
#define CondCore_CondDB_Logger_h
//
// Package:     CondDB
// Class  :     O2OLogger
//
/**\class Logger Logger.h CondCore/CondDB/interface/Logger.h
   Description: utility for collecting log information and store them into the Condition DB.  
*/
//
// Author:      Giacomo Govi
// Created:     Sep 2020
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include <sstream>
#include <memory>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class MsgDispatcher;

    template <typename EdmLogger>
    class EchoedLogStream {
    public:
      EchoedLogStream() = delete;
      explicit EchoedLogStream(const std::string& jobName, std::stringstream& buffer)
          : m_edmLogger(jobName), m_buffer(&buffer) {}
      virtual ~EchoedLogStream() {}
      template <typename T>
      EchoedLogStream& operator<<(const T& t) {
        *m_buffer << t;
        m_edmLogger << t;
        return *this;
      }
      EchoedLogStream& operator<<(std::ostream& (*f)(std::ostream&)) {
        *m_buffer << f;
        m_edmLogger << f;
        return *this;
      }
      EchoedLogStream& operator<<(std::ios_base& (*f)(std::ios_base&)) {
        *m_buffer << f;
        m_edmLogger << f;
        return *this;
      }

    private:
      EdmLogger m_edmLogger;
      std::stringstream* m_buffer;
    };

    template <>
    class EchoedLogStream<edm::LogDebug_> {
    public:
      EchoedLogStream() = delete;
      explicit EchoedLogStream(const std::string& jobName, std::stringstream& buffer)
          : m_edmLogger(jobName, __FILE__, __LINE__), m_buffer(&buffer) {}
      virtual ~EchoedLogStream() {}
      template <typename T>
      EchoedLogStream& operator<<(const T& t) {
        *m_buffer << t;
        m_edmLogger << t;
        return *this;
      }
      EchoedLogStream& operator<<(std::ostream& (*f)(std::ostream&)) {
        *m_buffer << f;
        m_edmLogger << f;
        return *this;
      }
      EchoedLogStream& operator<<(std::ios_base& (*f)(std::ios_base&)) {
        *m_buffer << f;
        m_edmLogger << f;
        return *this;
      }

    private:
      edm::LogDebug_ m_edmLogger;
      std::stringstream* m_buffer;
    };

    //
    class Logger {
    public:
      // default constructor is suppressed
      Logger() = delete;

      // constructor
      explicit Logger(const std::string& jobName);

      //
      virtual ~Logger();

      //
      void subscribeCoralMessages(const std::weak_ptr<MsgDispatcher>& dispatcher);

      //
      void setDbDestination(const std::string& connectionString);

      //
      void start();

      //
      void end(int retCode);

      //
      void saveOnDb();

      //
      void saveOnFile();

      //
      void save();

      //
      std::iostream& log(const std::string& tag);

      EchoedLogStream<edm::LogInfo> logInfo();
      EchoedLogStream<edm::LogDebug_> logDebug();
      EchoedLogStream<edm::LogError> logError();
      EchoedLogStream<edm::LogWarning> logWarning();

    private:
      void clearBuffer();

    private:
      std::string m_jobName;
      std::string m_connectionString;
      bool m_started;
      boost::posix_time::ptime m_startTime;
      boost::posix_time::ptime m_endTime;
      int m_retCode;
      std::stringstream m_log;
      std::weak_ptr<MsgDispatcher> m_dispatcher;
    };

  }  // namespace persistency
}  // namespace cond
#endif
