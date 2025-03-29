#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterHelpers_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterHelpers_h

#include <iomanip>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <typename T>
std::string prettyPrintVector(const std::vector<T>& data,
                              size_t head = 10,
                              size_t tail = 10,
                              size_t sample_points = 10,
                              size_t items_per_line = 10) {
  std::ostringstream out;
  const size_t n = data.size();

  auto print_block = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      out << std::setw(6) << data[i];
      if ((i - start + 1) % items_per_line == 0 || i == end - 1) {
        out << "\n";
      } else {
        out << " ";
      }
    }
  };

  if (n <= head + tail + sample_points) {
    print_block(0, n);
  } else {
    print_block(0, head);
    out << "...\n";

    for (size_t i = 1; i < sample_points - 1; ++i) {
      size_t idx = head + ((n - head - tail) * i) / (sample_points - 1);
      out << std::setw(6) << data[idx];
      if (i % items_per_line == 0 || i == sample_points - 2) {
        out << "\n";
      } else {
        out << " ";
      }
    }

    out << "...\n";
    print_block(n - tail, n);
  }

  return out.str();
}

class WarningSummary {
public:
  WarningSummary(const std::string& category, const std::string& name, bool debug = false)
      : m_debug(debug), m_category(category), m_name(name) {}

  inline void add(const std::string& message, const std::string& details = "") {
    const auto wIt = m_warnings.find(message);
    if (wIt == m_warnings.end()) {
      m_warnings.emplace(message, 1);
      if (m_debug) {
        // Removed warning at first err, as requested and accordingly to RecoLocalTracker/SiStripClusterizer/plugins/ClustersFromRawProducer.cc
        edm::LogWarning(m_category) << message << ": " << details
                                    << (m_debug
                                            ? ""
                                            : "\nNote: further warnings of this type will be suppressed (this can be "
                                              "changed by enabling debugging printout)");
      }
    } else {
      ++(wIt->second);
      if (m_debug) {
        edm::LogWarning(m_category) << message << ": " << details;
      }
    }
  }

  inline void printSummary() const {
    if (!m_warnings.empty()) {
      std::stringstream message;
      message << m_name << " warnings:";
      for (const auto& warnAndCount : m_warnings) {
        message << std::endl << warnAndCount.first << " (" << warnAndCount.second << ")";
      }
      edm::LogWarning(m_category) << message.str();
    }
  }

private:
  bool m_debug;
  std::string m_category;
  std::string m_name;
  std::map<std::string, std::size_t> m_warnings;
};

#endif
