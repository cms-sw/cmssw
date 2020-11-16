#include "CondCore/CondDB/interface/Logger.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Exception.h"
//
#include "DbCore.h"
#include "RelationalAccess/ITransaction.h"
//
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <fstream>
//
namespace cond {

  namespace persistency {

    conddb_table(O2O_RUN) {
      conddb_column(JOB_NAME, std::string);
      conddb_column(START_TIME, boost::posix_time::ptime);
      conddb_column(END_TIME, boost::posix_time::ptime);
      conddb_column(STATUS_CODE, int);
      conddb_column(LOG, std::string);
      class Table {
      public:
        explicit Table(coral::ISchema& schema) : m_schema(schema) {}
        ~Table() {}
        void insert(const std::string& jobName,
                    const boost::posix_time::ptime& start,
                    const boost::posix_time::ptime& end,
                    int retCode,
                    const std::string& log) {
          RowBuffer<JOB_NAME, START_TIME, END_TIME, STATUS_CODE, LOG> dataToInsert(
              std::tie(jobName, start, end, retCode, log));
          insertInTable(m_schema, tname, dataToInsert.get());
        }

      private:
        coral::ISchema& m_schema;
      };
    }

    void Logger::clearBuffer() {
      m_log.str("");
      m_log.clear();
    }

    Logger::Logger(const std::string& jobName)
        : m_jobName(jobName),
          m_connectionString(""),
          m_sharedConnectionPool(nullptr),
          m_started(false),
          m_startTime(),
          m_endTime(),
          m_retCode(0),
          m_log() {}

    //
    Logger::~Logger() {}

    void Logger::setDbDestination(const std::string& connectionString, ConnectionPool& connectionPool) {
      m_connectionString = connectionString;
      m_sharedConnectionPool = &connectionPool;
    }

    //
    void Logger::start() {
      if (!m_started) {
        if (!m_log.str().empty())
          clearBuffer();
        m_startTime = boost::posix_time::microsec_clock::universal_time();
        m_started = true;
        log("START_JOB") << " " << m_jobName;
      }
    }

    //
    void Logger::end(int retCode) {
      if (m_started) {
        m_endTime = boost::posix_time::microsec_clock::universal_time();
        m_started = false;
        m_retCode = retCode;
        log("END_JOB") << ": return code:" << retCode;
        save();
        clearBuffer();
      }
    }

    std::string print_timestamp(const boost::posix_time::ptime& t, const char* format_s = "%Y-%m-%d %H:%M:%S.%f") {
      boost::posix_time::time_facet* facet = new boost::posix_time::time_facet();
      facet->format(format_s);
      std::stringstream timestamp;
      timestamp.imbue(std::locale(std::locale::classic(), facet));
      timestamp << t;
      return timestamp.str();
    }

    std::string get_timestamp() {
      auto now = boost::posix_time::microsec_clock::universal_time();
      return print_timestamp(now);
    }
    std::string get_timestamp_for_filename() {
      auto now = boost::posix_time::microsec_clock::universal_time();
      return print_timestamp(now, "%Y-%m-%d_%H-%M-%S");
    }

    //
    void Logger::saveOnFile() {
      if (!m_log.str().empty()) {
        std::string fileName(get_timestamp_for_filename() + ".log");
        std::ofstream fout(fileName, std::ofstream::app);
        fout << m_log.str() << std::endl;
        fout.close();
      }
    }

    //
    void Logger::saveOnDb() {
      if (!m_log.str().empty()) {
        if (m_sharedConnectionPool == nullptr) {
          throwException("Connection pool handle has not been provided.", "Logger::saveOnDb");
        }
        if (m_connectionString.empty()) {
          throwException("Connection string for destination database has not been provided.", "Logger::saveOnDb");
        }
        auto coralSession = m_sharedConnectionPool->createCoralSession(m_connectionString, true);
        coralSession->transaction().start(false);
        try {
          O2O_RUN::Table destinationTable(coralSession->nominalSchema());
          destinationTable.insert(m_jobName, m_startTime, m_endTime, m_retCode, m_log.str());
          coralSession->transaction().commit();
        } catch (const std::exception& e) {
          coralSession->transaction().rollback();
          // dump on file on this circumstance...
          logError() << e.what();
          saveOnFile();
          throwException(std::string("Failure while saving log on database:") + e.what(), "Logger::saveOnDb");
        }
      }
    }

    void Logger::save() {
      if (!m_connectionString.empty())
        saveOnDb();
      else
        saveOnFile();
    }

    std::iostream& Logger::log(const std::string& tag) {
      if (std::size(m_log.str()) != 0)
        m_log << std::endl;
      m_log << "[" << get_timestamp() << "] " << tag << ": ";
      return m_log;
    }

    EchoedLogStream<edm::LogInfo> Logger::logInfo() {
      log("INFO");
      return EchoedLogStream<edm::LogInfo>(m_jobName, m_log);
    }
    EchoedLogStream<edm::LogDebug_> Logger::logDebug() {
      log("DEBUG");
      return EchoedLogStream<edm::LogDebug_>(m_jobName, m_log);
    }
    EchoedLogStream<edm::LogError> Logger::logError() {
      log("ERROR");
      return EchoedLogStream<edm::LogError>(m_jobName, m_log);
    }
    EchoedLogStream<edm::LogWarning> Logger::logWarning() {
      log("WARNING");
      return EchoedLogStream<edm::LogWarning>(m_jobName, m_log);
    }

  }  // namespace persistency
}  // namespace cond
