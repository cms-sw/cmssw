
#ifndef Utilities_StorageFactory_StatisticsSenderService_H
#define Utilities_StorageFactory_StatisticsSenderService_H

#include <string>
#include <sstream>
#include <atomic>
#include <mutex>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include "FWCore/Utilities/interface/InputType.h"

namespace edm {

  class ParameterSet;
  class ActivityRegistry;

  namespace storage {

    class StatisticsSenderService {
    public:
      StatisticsSenderService(edm::ParameterSet const& pset, edm::ActivityRegistry& ar);

      void setSize(const std::string& urlOrLfn, size_t size);
      void setCurrentServer(const std::string& urlOrLfn, const std::string& servername);
      static const char* getJobID();
      static bool getX509Subject(std::string&);

      void openingFile(std::string const& lfn, edm::InputType type, size_t size = -1);
      void closedFile(std::string const& lfn, bool usedFallback);

    private:
      void filePostCloseEvent(std::string const& lfn);

      std::string const* matchedLfn(std::string const& iURL);  //updates its internal cache
      class FileStatistics {
      public:
        FileStatistics();
        void fillUDP(std::ostringstream& os) const;
        void update();

      private:
        ssize_t m_read_single_operations;
        ssize_t m_read_single_bytes;
        ssize_t m_read_single_square;
        ssize_t m_read_vector_operations;
        ssize_t m_read_vector_bytes;
        ssize_t m_read_vector_square;
        ssize_t m_read_vector_count_sum;
        ssize_t m_read_vector_count_square;
        time_t m_start_time;
      };

      struct FileInfo {
        explicit FileInfo(std::string const& iLFN, edm::InputType);

        FileInfo(FileInfo&& iInfo)
            : m_filelfn(std::move(iInfo.m_filelfn)),
              m_serverhost(std::move(iInfo.m_serverhost)),
              m_serverdomain(std::move(iInfo.m_serverdomain)),
              m_type(iInfo.m_type),
              m_size(iInfo.m_size.load()),
              m_id(iInfo.m_id),
              m_openCount(iInfo.m_openCount.load()) {}
        std::string m_filelfn;
        std::string m_serverhost;
        std::string m_serverdomain;
        edm::InputType m_type;
        std::atomic<ssize_t> m_size;
        size_t m_id;  //from m_counter
        std::atomic<int> m_openCount;
      };

      void determineHostnames();
      void fillUDP(const std::string& site, const FileInfo& fileinfo, bool, std::string&) const;
      void cleanupOldFiles();

      std::string m_clienthost;
      std::string m_clientdomain;
      oneapi::tbb::concurrent_unordered_map<std::string, FileInfo> m_lfnToFileInfo;
      oneapi::tbb::concurrent_unordered_map<std::string, std::string> m_urlToLfn;
      FileStatistics m_filestats;
      std::string m_guid;
      size_t m_counter;
      std::string m_userdn;
      std::mutex m_servermutex;
      const bool m_debug;
    };

  }  // namespace storage
}  // namespace edm

#endif
