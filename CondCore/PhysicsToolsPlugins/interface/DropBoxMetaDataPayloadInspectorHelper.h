#ifndef DropBoxMetaDataPayloadInspectorHelper_H
#define DropBoxMetaDataPayloadInspectorHelper_H

#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <fmt/printf.h>
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include "CondFormats/Common/interface/DropBoxMetadata.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace DPMetaDataHelper {
  class RecordMetaDataInfo {
  public:
    /// Constructor
    RecordMetaDataInfo(DropBoxMetadata::Parameters params) {
      const auto& theParameters = params.getParameterMap();
      for (const auto& [key, val] : theParameters) {
        if (key.find("prep") != std::string::npos) {
          m_prepmetadata = val;
        } else if (key.find("prod") != std::string::npos) {
          m_prodmetadata = val;
        } else if (key.find("mult") != std::string::npos) {
          m_multimetadata = val;
        }
      }
    }
    /// Destructor
    ~RecordMetaDataInfo() = default;

  public:
    const std::string getPrepMetaData() const { return m_prepmetadata; }
    const std::string getProdMetaData() const { return m_prodmetadata; }
    const std::string getMultiMetaData() const { return m_multimetadata; }

  private:
    std::string m_prepmetadata;
    std::string m_prodmetadata;
    std::string m_multimetadata;
  };

  using recordMap = std::map<std::string, RecordMetaDataInfo>;

  class DBMetaDataTableDisplay {
  public:
    DBMetaDataTableDisplay(DPMetaDataHelper::recordMap theMap) : m_Map(theMap) {}
    ~DBMetaDataTableDisplay() = default;

    void printMetaDatas() {
      for (const auto& [key, val] : m_Map) {
        std::cout << "key:" << key << "\n\n" << std::endl;
        std::cout << "prep:" << replaceAll(val.getPrepMetaData(), std::string("&quot;"), std::string("'")) << "\n"
                  << std::endl;
        std::cout << "prod:" << replaceAll(val.getProdMetaData(), std::string("&quot;"), std::string("'")) << "\n"
                  << std::endl;
        std::cout << "multi:" << replaceAll(val.getMultiMetaData(), std::string("&quot;"), std::string("'")) << "\n"
                  << std::endl;
      }
    }

  private:
    DPMetaDataHelper::recordMap m_Map;
    std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
      size_t start_pos = 0;
      while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // Handles case where 'to' is a substring of 'from'
      }
      return str;
    }
  };
}  // namespace DPMetaDataHelper
#endif
