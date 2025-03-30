#ifndef RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterHelpers_h
#define RecoLocalTracker_SiStripClusterizer_plugins_alpaka_SiStripRawToClusterHelpers_h

#include <iomanip>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

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

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;
  class DataFedAppender_new {
  public:
    DataFedAppender_new(Queue& queue, unsigned int bufferSize)
        : bufferSize_byte_(bufferSize / sizeof(uint8_t)),
          fedIDinSet_(sistrip::NUMBER_OF_FEDS, false),
          bufferData_(cms::alpakatools::make_host_buffer<uint8_t[]>(queue, bufferSize_byte_)),
          offset_(0) {};

    void insertData(uint16_t fedID, const FEDRawData* rawFEDData) {
      if (rawFEDData->data() == nullptr) {
        edm::LogWarning("DataFedAppender_new") << "FED " << fedID << " has no data";
        return;
      } else if (rawFEDData->size() == 0) {
        edm::LogWarning("DataFedAppender_new") << "FED " << fedID << " has empty data";
        return;
      } else {
        // Copy the data into pinned memory
        std::memcpy(bufferData_.data() + offset_, rawFEDData->data(), rawFEDData->size());
        offset_ += rawFEDData->size();
        fedIDinSet_[fedID - sistrip::FED_ID_MIN] = true;

        validFED_offsets[fedID] = offset_;
        chunkStartIdx_.emplace_back(offset_);
      }
    }

    auto getData() const { return bufferData_; }
    inline size_t getOffset(uint16_t fedID) { return validFED_offsets[fedID]; }
    inline auto bufferSize_byte() { return bufferSize_byte_; }

    // Is the fedID in the set?
    bool isInside(uint16_t fedID) const {
      uint16_t fedi = fedID - sistrip::FED_ID_MIN;
      if (fedi < fedIDinSet_.size())
        return fedIDinSet_[fedi];
      return false;
    }

  private:
    const unsigned int bufferSize_byte_;
    std::vector<bool> fedIDinSet_;

    cms::alpakatools::host_buffer<uint8_t[]> bufferData_;
    unsigned int offset_;

    std::map<uint16_t, size_t> validFED_offsets;
    std::vector<unsigned int> chunkStartIdx_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#endif
