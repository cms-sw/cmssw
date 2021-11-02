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

namespace DBoxMetadataHelper {
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
    const bool hasMultiMetaData() const { return !m_multimetadata.empty(); }

  private:
    std::string m_prepmetadata;
    std::string m_prodmetadata;
    std::string m_multimetadata;
  };

  using recordMap = std::map<std::string, RecordMetaDataInfo>;

  inline const std::vector<std::string> getAllRecords(const DBoxMetadataHelper::recordMap& recordSet) {
    std::vector<std::string> records;
    std::transform(recordSet.begin(),
                   recordSet.end(),
                   std::inserter(records, records.end()),
                   [](std::pair<std::string, DBoxMetadataHelper::RecordMetaDataInfo> recordSetEntry) -> std::string {
                     return recordSetEntry.first;
                   });
    return records;
  }

  inline std::vector<std::string> set_difference(std::vector<std::string> const& v1,
                                                 std::vector<std::string> const& v2) {
    std::vector<std::string> diff;
    std::set_difference(std::begin(v1), std::end(v1), std::begin(v2), std::end(v2), std::back_inserter(diff));
    return diff;
  }

  inline std::vector<std::string> set_intersection(std::vector<std::string> const& v1,
                                                   std::vector<std::string> const& v2) {
    std::vector<std::string> common;
    std::set_intersection(std::begin(v1), std::end(v1), std::begin(v2), std::end(v2), std::back_inserter(common));
    return common;
  }

  class DBMetaDataTableDisplay {
  public:
    DBMetaDataTableDisplay(DBoxMetadataHelper::recordMap theMap) : m_Map(theMap) {}
    ~DBMetaDataTableDisplay() = default;

    void printMetaDatas() {
      for (const auto& [key, val] : m_Map) {
        edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << key << "\n\n" << std::endl;
        edm::LogPrint("DropBoxMetadataPIHelper") << "prep: " << cleanJson(val.getPrepMetaData()) << "\n" << std::endl;
        edm::LogPrint("DropBoxMetadataPIHelper") << "prod: " << cleanJson(val.getProdMetaData()) << "\n" << std::endl;
        // check, since it's optional
        if (val.hasMultiMetaData()) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "multi: " << cleanJson(val.getMultiMetaData()) << "\n"
                                                   << std::endl;
        }
      }
    }

    void printOneKey(const DBoxMetadataHelper::RecordMetaDataInfo& oneKey) {
      edm::LogPrint("DropBoxMetadataPIHelper") << "prep: " << cleanJson(oneKey.getPrepMetaData()) << std::endl;
      edm::LogPrint("DropBoxMetadataPIHelper") << "prod: " << cleanJson(oneKey.getProdMetaData()) << std::endl;
      // check, since it's optional
      if (oneKey.hasMultiMetaData()) {
        edm::LogPrint("DropBoxMetadataPIHelper") << "multi: " << cleanJson(oneKey.getMultiMetaData()) << std::endl;
      }
      edm::LogPrint("DropBoxMetadataPIHelper") << "\n" << std::endl;
    }

    void printDiffWithMetadata(const DBoxMetadataHelper::recordMap& theRefMap) {
      edm::LogPrint("DropBoxMetadataPIHelper")
          << "Target has: " << m_Map.size() << " records, reference has: " << theRefMap.size() << " records"
          << std::endl;

      const auto& ref_records = DBoxMetadataHelper::getAllRecords(theRefMap);
      const auto& tar_records = DBoxMetadataHelper::getAllRecords(m_Map);

      const auto& diff = DBoxMetadataHelper::set_difference(ref_records, tar_records);
      const auto& common = DBoxMetadataHelper::set_intersection(ref_records, tar_records);

      // do first the common parts
      for (const auto& key : common) {
        edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << key << "\n" << std::endl;
        const auto& val = m_Map.at(key);
        const auto& refval = theRefMap.at(key);

        if ((val.getPrepMetaData()).compare(refval.getPrepMetaData()) != 0) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "found difference in prep metadata!" << std::endl;
          edm::LogPrint("DropBoxMetadataPIHelper") << " in target: " << cleanJson(val.getPrepMetaData()) << std::endl;
          edm::LogPrint("DropBoxMetadataPIHelper")
              << " in reference: " << cleanJson(refval.getPrepMetaData()) << std::endl;
        }
        if ((val.getProdMetaData()).compare(refval.getProdMetaData()) != 0) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "found difference in prod metadata!" << std::endl;
          edm::LogPrint("DropBoxMetadataPIHelper") << " in target: " << cleanJson(val.getProdMetaData()) << std::endl;
          edm::LogPrint("DropBoxMetadataPIHelper")
              << " in reference: " << cleanJson(refval.getProdMetaData()) << std::endl;
        }
        if ((val.getMultiMetaData()).compare(refval.getMultiMetaData()) != 0) {
          edm::LogPrint("DropBoxMetadataPIHelper") << "found difference in multi metadata!" << std::endl;
          edm::LogPrint("DropBoxMetadataPIHelper") << " in target: " << cleanJson(val.getMultiMetaData()) << std::endl;
          edm::LogPrint("DropBoxMetadataPIHelper")
              << " in reference: " << cleanJson(refval.getMultiMetaData()) << std::endl;
        }
        edm::LogPrint("DropBoxMetadataPIHelper") << "\n" << std::endl;
      }

      // if interesction is not the union check for extra differences
      if (!diff.empty()) {
        // check if the reference has more records than target
        if (ref_records.size() > tar_records.size()) {
          for (const auto& ref : ref_records) {
            if (std::find(tar_records.begin(), tar_records.end(), ref) == tar_records.end()) {
              const auto& refval = theRefMap.at(ref);
              edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << ref << " not present in target! \n" << std::endl;
              printOneKey(refval);
            }
          }
        }
        // then check if the target has more records than the reference
        else if (tar_records.size() > ref_records.size()) {
          for (const auto& tar : tar_records) {
            if (std::find(ref_records.begin(), ref_records.end(), tar) == ref_records.end()) {
              const auto& tarval = m_Map.at(tar);
              edm::LogPrint("DropBoxMetadataPIHelper") << "key: " << tar << " not present in reference! \n"
                                                       << std::endl;
              printOneKey(tarval);
            }
          }
        }
      }
    }

  private:
    DBoxMetadataHelper::recordMap m_Map;

    std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
      size_t start_pos = 0;
      while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // Handles case where 'to' is a substring of 'from'
      }
      return str;
    }

    std::string cleanJson(std::string str) {
      std::string out = replaceAll(str, std::string("&quot;"), std::string("'"));
      return out;
    }
  };
}  // namespace DBoxMetadataHelper
#endif
