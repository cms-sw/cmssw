#include "DQM/HGCAL/interface/HGCalDQMCommon.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoSoA.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <nlohmann/json.hpp>
#include "DQMServices/Core/interface/MonitorElement.h"

#include <TFile.h>
#include <TTree.h>

namespace hgcal {
  namespace dqm {

    //
    std::string getLabelForSummaryIndex(SummaryIndices_t idx) {
      std::string label = "<CM>";
      if (idx == SummaryIndices_t::PEDESTAL)
        label = "Pedestal";
      else if (idx == SummaryIndices_t::NOISE)
        label = "Noise";
      else if (idx == SummaryIndices_t::DELTAPEDESTAL)
        label = "#DeltaPedestal";
      else if (idx == SummaryIndices_t::TOAAVG)
        label = "<TOA>";
      else if (idx == SummaryIndices_t::TOTAVG)
        label = "<TOT>";
      return label;
    }

    //--------------------------------------------------
    // Common helper functions
    //--------------------------------------------------
    void addBinLabels(std::vector<std::string>& binlabels, MonitorElement* hist, int binPos) {
      for (size_t i = 0; i < binlabels.size(); i++) {
        hist->setBinLabel(i + 1, binlabels[i], binPos);
      }
    }

    //--------------------------------------------------
    // ECON-D flags
    //--------------------------------------------------
    std::vector<std::string> econdWithCBflags = {
        // ECOND flags
        "H/T good",
        "H/T fail",
        "H/T amb",
        "E/B/O good",
        "E/B/O fail",
        "E/B/O amb",
        "Unmatched (M)",
        "Trunc (T)",
        "Unexpected (E)",
        "Sub-packet error (S)",
        "Marker",
        "Payload (OF)",
        "Payload (mismatch)",

        // ---- CB flags below ----
        "CB: !Normal",
        "Payload",
        "CRC Error",
        "EvID Mis.",
        "FSM T/O",
        "BCID/OrbitID",
        "MB Overflow",
        "Inactive",
        "CRC trailer error"};

    //--------------------------------------------------
    // Assigns the bins to filled with ECON-D flags
    //--------------------------------------------------
    std::vector<int> getErrorBinsForECONDCBFlags(unsigned int cbflag, unsigned int econdflag, unsigned int exception) {
      std::vector<int> errorBin;
      constexpr size_t necondflags = 13;
      constexpr size_t ncbflags = 8;

      // ECON-D quality flags
      auto htflags = hgcaldigi::htFlag(econdflag);
      auto eboflags = hgcaldigi::eboFlag(econdflag);
      if (htflags > 0)
        errorBin.push_back(0 + htflags);  //1, 2, 3 bins
      if (eboflags > 0)
        errorBin.push_back(3 + eboflags);  //4, 5, 6 bins
      if (hgcaldigi::matchFlag(econdflag) == 0)
        errorBin.push_back(7);
      if (hgcaldigi::truncatedFlag(econdflag))
        errorBin.push_back(8);
      if (hgcaldigi::expectedFlag(econdflag) == 0)
        errorBin.push_back(9);
      if (hgcaldigi::StatFlag(econdflag) == 1)
        errorBin.push_back(10);
      if (exception == 3)
        errorBin.push_back(11);  // wrongHeaderMarker
      if (exception == 4)
        errorBin.push_back(12);  // payloadOverflows
      if (exception == 5)
        errorBin.push_back(13);  // payloadMismatches

      // CB quality flags
      if (cbflag > 0) {
        unsigned int purecb_flag = cbflag & 0x7;
        bool crctrailer_error = (cbflag >> 3) & 0x1;

        if (purecb_flag > 0)
          errorBin.push_back(necondflags + purecb_flag);  //14-21 bins

        if (crctrailer_error)
          errorBin.push_back(necondflags + ncbflags + 1);  //22nd bin
      }

      return errorBin;
    }

    std::string econdErrorTypeToString(EcondErrorType type) {
      if (static_cast<size_t>(type) >= econdWithCBflags.size()) {
        return "Unknown";
      }
      return econdWithCBflags[static_cast<size_t>(type)];
    }

    EcondErrorType stringToEcondErrorType(const std::string& errorString) {
      for (size_t i = 0; i < econdWithCBflags.size(); ++i) {
        if (econdWithCBflags[i] == errorString) {
          return static_cast<EcondErrorType>(i);
        }
      }
      return EcondErrorType::NUM_ERROR_TYPES;  // Invalid/not found
    }

    // Category names for labeling
    std::vector<std::string> qualityCategoryNames = {
        "CB Issues",       // CB_ISSUES = 0
        "ECON-D Payload",  // ECOND_PAYLOAD = 1
        "Header/Trailer"   // HEADER_TRAILER = 2
    };

    // Mapping from ECON-D/CB flags to quality categories
    std::map<EcondErrorType, ErrorCategory> errorTypeToCategory = {
        // CB Issues (capture block associated problems)
        {EcondErrorType::CB_NOT_NORMAL, ErrorCategory::CB_ISSUES},
        {EcondErrorType::CB_PAYLOAD, ErrorCategory::CB_ISSUES},
        {EcondErrorType::CRC_ERROR, ErrorCategory::CB_ISSUES},
        {EcondErrorType::EVID_MISMATCH, ErrorCategory::CB_ISSUES},
        {EcondErrorType::FSM_TIMEOUT, ErrorCategory::CB_ISSUES},
        {EcondErrorType::BCID_ORBITID, ErrorCategory::CB_ISSUES},
        {EcondErrorType::MB_OVERFLOW, ErrorCategory::CB_ISSUES},
        {EcondErrorType::INACTIVE, ErrorCategory::CB_ISSUES},

        // ECON-D Data (endcap concentrator data packet problems)

        {EcondErrorType::EBO_GOOD, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::EBO_FAIL, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::EBO_AMB, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::UNMATCHED, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::TRUNC, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::UNEXPECTED, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::SUB_PACKET_ERROR, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::MARKER, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::PAYLOAD_OF, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::PAYLOAD_MISMATCH, ErrorCategory::ECOND_PAYLOAD},
        {EcondErrorType::CRC_TRAILER_ERROR, ErrorCategory::ECOND_PAYLOAD},

        // Header/Trailer (basic data packet format)
        {EcondErrorType::HT_GOOD, ErrorCategory::HEADER_TRAILER},
        {EcondErrorType::HT_FAIL, ErrorCategory::HEADER_TRAILER},
        {EcondErrorType::HT_AMB, ErrorCategory::HEADER_TRAILER}};

    std::vector<std::string> econdTFlags = {"nTC not matching (BC)", "Subpacket Error (S)", "TDaqIdx Out Range"};

    std::map<EconTErrorType, EconTErrorCategory> econdTErrorTypeToCategory = {
        {EconTErrorType::NTC_NOT_MATCHING, EconTErrorCategory::UNPACKING_ERRORS},
        {EconTErrorType::SUBPACKET_ERROR, EconTErrorCategory::HEADER_TRAILER},
        {EconTErrorType::TDAQIDX_OUT_RANGE, EconTErrorCategory::UNPACKING_ERRORS}};

    std::vector<std::string> econtQualityCategoryNames = {"Unpacking Errors", "Header/Trailer"};

    std::string econdTErrorTypeToString(EconTErrorType type) {
      if (static_cast<size_t>(type) >= econdTFlags.size()) {
        return "Unknown";
      }
      return econdTFlags[static_cast<size_t>(type)];
    }

    EconTErrorType stringToEconTErrorType(const std::string& errorString) {
      for (size_t i = 0; i < econdTFlags.size(); ++i) {
        if (econdTFlags[i] == errorString) {
          return static_cast<EconTErrorType>(i);
        }
      }
      return EconTErrorType::NUM_ERROR_TYPES;
    }

    //--------------------------------------------------
    // Utility functions in Report structure
    //--------------------------------------------------
    void Report::addError(EcondErrorType type, int count, CategoryID category) {
      categorized_counters[category][static_cast<size_t>(type)] += count;
    }

    int Report::getCategoryErrorCount(CategoryID category, EcondErrorType type) const {
      auto it = categorized_counters.find(category);
      if (it != categorized_counters.end()) {
        return it->second[static_cast<size_t>(type)];
      }
      return 0;
    }

    int Report::getCategoryTotalErrors(CategoryID category) const {
      int total = 0;
      auto it = categorized_counters.find(category);
      if (it != categorized_counters.end()) {
        for (const auto& count : it->second) {
          total += count;
        }
      }
      return total;
    }

    std::string Report::getCategoryErrorSummary(CategoryID category) const {
      std::stringstream ss;
      auto it = categorized_counters.find(category);
      if (it != categorized_counters.end()) {
        for (size_t i = 0; i < static_cast<size_t>(EcondErrorType::NUM_ERROR_TYPES); ++i) {
          if (it->second[i] > 0) {
            ss << econdWithCBflags[i] << ": " << it->second[i] << ", ";
          }
        }
      }
      std::string result = ss.str();
      if (!result.empty()) {
        result = result.substr(0, result.length() - 2);  // Remove trailing comma and space
      }
      return result;
    }

    std::vector<CategoryID> Report::getAllCategories() const {
      std::vector<CategoryID> categories;
      categories.reserve(categorized_counters.size());
      for (const auto& pair : categorized_counters) {
        categories.push_back(pair.first);
      }
      return categories;
    }

    //--------------------------------------------------
    // Methods in summarizer
    //--------------------------------------------------
    EcondErrorSummarizer::EcondErrorSummarizer() : config_data_(json{}) {}

    EcondErrorSummarizer::EcondErrorSummarizer(const json& config_data) : config_data_(config_data) {
      loadMatrixConfiguration();
    }

    void EcondErrorSummarizer::loadMatrixConfiguration() {
      try {
        if (config_data_.contains("error_thresholds_matrix")) {
          auto matrix_json = config_data_["error_thresholds_matrix"];
          error_thresholds_matrix_.resize(matrix_json.size());

          for (size_t i = 0; i < matrix_json.size(); ++i) {
            error_thresholds_matrix_[i] = matrix_json[i].get<std::vector<int>>();
          }
        }

        if (config_data_.contains("error_type_labels")) {
          error_type_labels_ = config_data_["error_type_labels"].get<std::vector<std::string>>();
        }

        if (error_thresholds_matrix_.size() != error_type_labels_.size()) {
          edm::LogWarning("HGCalDQMEcondErrorSummarizer") << "Matrix size (" << error_thresholds_matrix_.size()
                                                          << ") != Labels size (" << error_type_labels_.size() << ")";
        }

      } catch (const std::exception& e) {
        edm::LogError("HGCalDQMEcondErrorSummarizer") << "Error loading matrix configuration: " << e.what();
      }
    }

    std::vector<double> EcondErrorSummarizer::sumAxis(MonitorElement* me, int axis) {
      int nbinsX = me->getNbinsX();
      int nbinsY = me->getNbinsY();
      std::vector<double> sums;
      if (axis == 1) {
        for (int y = 0; y < nbinsY; ++y) {  // y axis
          double sum = 0;
          for (int x = 0; x < nbinsX; ++x) {
            sum += me->getBinContent(x + 1, y + 1);
          }
          LogDebug("HGCalDQMEcondErrorSummarizer") << "ysum " << sum;
          sums.push_back(sum);
        }
      } else if (axis == 2) {  // x-axis
        for (int x = 0; x < nbinsX; ++x) {
          double sum = 0;
          for (int y = 0; y < nbinsY; ++y) {
            sum += me->getBinContent(x + 1, y + 1);
          }
          sums.push_back(sum);
        }
      } else {
        sums.clear();
      }
      return sums;
    }

    void EcondErrorSummarizer::processAndFillLS(MonitorElement* source_quality_layer,
                                                int binNumber,
                                                MonitorElement* target_quality_LS,
                                                MonitorElement* target_finequality_LS,
                                                MonitorElement* target_layer_LS) {
      // This is the raw sums for EconD versus LS
      processAndFill(source_quality_layer, target_finequality_LS, binNumber, ProcessMode::STAT_TO_STAT);
      // This is EconD quality versus LS
      processAndFill(source_quality_layer, target_quality_LS, binNumber, ProcessMode::STAT_TO_GRADE);
      // This is layer quality versus LS
      processAndFill(source_quality_layer, target_layer_LS, binNumber, ProcessMode::STAT_TO_GRADE_X);
    }

    // accumulate stats from srouce and fill the total stat into a specified bin of target monitor element
    void EcondErrorSummarizer::processAndFill(const MonitorElement* source_me,
                                              MonitorElement* target_me,
                                              const int target_bin_id,
                                              const ProcessMode mode) {
      switch (mode) {
        case ProcessMode::STAT_TO_STAT:
          for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
            double total = 0;
            for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
              total += source_me->getBinContent(xbin, ybin);
            }
            if (total > 0) {
              target_me->setBinContent(target_bin_id + 1, ybin, total);
            }
          }
          break;

        case ProcessMode::STAT_TO_GRADE:  // also from EcondErrorType to ErrorCategory
        {
          std::map<ErrorCategory, int> worst_grades = {
              {ErrorCategory::CB_ISSUES, 0}, {ErrorCategory::ECOND_PAYLOAD, 0}, {ErrorCategory::HEADER_TRAILER, 0}};

          // Find worst quality grade for modules in a cassette
          for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
            int error_type_index = ybin - 1;
            if (!(error_type_index < static_cast<int>(EcondErrorType::NUM_ERROR_TYPES)))
              continue;
            ErrorCategory category = errorTypeToCategory[static_cast<EcondErrorType>(error_type_index)];

            for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
              double content = source_me->getBinContent(xbin, ybin);
              int current_grade = calculateGrade(error_type_index, static_cast<int>(content));
              worst_grades[category] = std::max(worst_grades[category], current_grade);
            }
          }

          // Fill worst grade values based on ErrorCategory
          for (size_t ybin = 1; ybin <= static_cast<size_t>(ErrorCategory::NUM_CATEGORIES); ++ybin) {
            ErrorCategory error_type_category = static_cast<ErrorCategory>(ybin - 1);
            target_me->setBinContent(target_bin_id + 1, ybin, worst_grades[error_type_category]);
          }
        } break;

        case ProcessMode::GRADE_TO_GRADE:  // ErrorCategory to ErrorCategory
          for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
            int worst_grade = 0;
            for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
              int grade = source_me->getBinContent(xbin, ybin);
              worst_grade = std::max(worst_grade, grade);
            }
            target_me->setBinContent(target_bin_id + 1, ybin, worst_grade);
          }
          break;

        case ProcessMode::STAT_TO_GRADE_X:  // also from EcondErrorType to ErrorCategory
        {
          std::vector<int> worst_grades;  // will have a worst grade for each layer (x-value).
          // Find worst quality grade for modules in a cassette
          for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
            int worst_grade = 0;
            std::string worst_grade_name;
            for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
              int error_type_index = ybin - 1;
              if (!(error_type_index < static_cast<int>(EcondErrorType::NUM_ERROR_TYPES)))
                continue;
              //ErrorCategory category = errorTypeToCategory[static_cast<EcondErrorType>(error_type_index)];
              double content = source_me->getBinContent(xbin, ybin);
              int current_grade = calculateGrade(error_type_index, static_cast<int>(content));
              worst_grade = std::max(worst_grade, current_grade);
            }
            worst_grades.push_back(worst_grade);
          }

          // Fill worst grade values based on ErrorCategory
          for (size_t ybin = 1; ybin <= worst_grades.size(); ++ybin) {
            //ErrorCategory error_type_category = static_cast<ErrorCategory>(ybin-1);
            target_me->setBinContent(target_bin_id + 1, ybin, worst_grades[ybin - 1]);
          }
        } break;

        default:
          edm::LogError("HGCalDQMEcondErrorSummarizer") << "Invalid mode " << static_cast<int>(mode);
          break;
      }
    }

    // Calculate quality grade (1=best, 5=worst) based on error count for given flag
    int EcondErrorSummarizer::calculateGrade(int error_type_index, int num_errors) {
      if (error_type_index < 0 || error_type_index >= static_cast<int>(error_thresholds_matrix_.size())) {
        edm::LogError("HGCalDQMEcondErrorSummarizer") << "Invalid error type index " << error_type_index;
        return 5;
      }

      const auto& thresholds = error_thresholds_matrix_[error_type_index];
      for (size_t grade = 0; grade < thresholds.size(); ++grade) {
        if (num_errors <= thresholds[grade])
          return static_cast<int>(grade + 1);
      }

      return 5;  // return worst grade
    }

    // checking channel stats of a polygonal histogram
    std::map<std::string, int> EcondErrorSummarizer::analyzeChannelQuality(MonitorElement* me, double threshold) {
      std::map<std::string, int> stats = {{"noisy", 0}, {"stuck", 0}, {"normal", 0}, {"total", 0}};

      if (!me)
        return stats;

      if (me->kind() == MonitorElement::Kind::TH2Poly) {
        TH2Poly* hist = me->getTH2Poly();
        stats["total"] = hist->GetNumberOfBins();

        for (int bin = 1; bin <= stats["total"]; ++bin) {
          double content = hist->GetBinContent(bin);

          // here we expect to process the polygonal map of `stdadc`
          // content here means `standard deviation`
          // and a zero stdadc indicates the channel is stuck
          if (content == 0.0) {
            stats["stuck"]++;
          } else if (content > threshold) {
            stats["noisy"]++;
          } else {
            stats["normal"]++;
          }
        }
      }
      assert(stats["noisy"] + stats["stuck"] + stats["normal"] == stats["total"]);
      return stats;
    }

    // Instance method for config-based threshold
    std::map<std::string, int> EcondErrorSummarizer::analyzeChannelQuality(MonitorElement* me,
                                                                           const std::string& threshold_key) {
      try {
        double threshold = config_data_[threshold_key];
        return analyzeChannelQuality(me, threshold);
      } catch (const std::exception& e) {
        return {{"noisy", 0}, {"stuck", 0}, {"normal", 0}, {"total", 0}};
      }
    }

    std::map<std::string, int> EcondErrorSummarizer::count_zero_std_bins(const TProfile* profile, int nMax) {
      std::map<std::string, int> stats = {{"vacant", 0}, {"stuck", 0}};
      for (int i = 1; i <= nMax; i++) {
        double content = profile->GetBinContent(i);
        if (content == 0.0) {
          stats["vacant"]++;
        }
        double binError = profile->GetBinError(i);
        if (binError == 0.0) {
          stats["stuck"]++;
        }
      }
      return stats;
    }
    //--------------------------------------------------

    Report EcondErrorSummarizer::aggregateReports(const std::vector<Report>& reports) {
      Report aggregatedReport;

      // Aggregate all categorized counters from all reports
      for (const auto& report : reports) {
        for (const auto& categoryPair : report.categorized_counters) {
          CategoryID category = categoryPair.first;
          const auto& errorCounts = categoryPair.second;

          // Add this category's error counts to the aggregated report
          for (size_t i = 0; i < static_cast<size_t>(EcondErrorType::NUM_ERROR_TYPES); ++i) {
            aggregatedReport.categorized_counters[category][i] += errorCounts[i];
          }
        }
      }

      return aggregatedReport;
    }

    Report EcondErrorSummarizer::analysisWithCategories(MonitorElement* me,
                                                        const std::function<CategoryID(int)>& categorizer) {
      Report output;

      if (!me || me->kind() != MonitorElement::Kind::TH2F) {
        return output;
      }

      // Get the TH2F from the MonitorElement
      TH2F* hist = me->getTH2F();
      if (!hist)
        return output;

      // Iterate over all bins in the histogram
      for (int xbin = 1; xbin <= hist->GetNbinsX(); ++xbin) {
        // Determine the category for this x-bin
        CategoryID category = categorizer(xbin);

        for (int ybin = 1; ybin <= hist->GetNbinsY(); ++ybin) {
          double content = hist->GetBinContent(xbin, ybin);
          if (content > 0) {
            EcondErrorType errorType = static_cast<EcondErrorType>(ybin - 1);
            output.addError(errorType, static_cast<int>(content), category);
          }
        }
      }

      return output;
    }

    void EcondErrorSummarizer::fillHistogramFromCategoryReport(MonitorElement* me,
                                                               const Report& report,
                                                               CategoryID category) {
      // Ensure the monitor element is a TH2F
      if (!me || me->kind() != MonitorElement::Kind::TH2F) {
        edm::LogError("HGCalDQMEcondErrorSummarizer") << "MonitorElement is not a TH2F histogram";
        return;
      }

      // Fill the histogram with values from the category in the report
      for (size_t iType = 0; iType < static_cast<size_t>(EcondErrorType::NUM_ERROR_TYPES); iType++) {
        int value = report.getCategoryErrorCount(category, static_cast<EcondErrorType>(iType));

        if (value > 0) {                                      // Only set non-zero values
          me->setBinContent(category + 1, iType + 1, value);  // +1 because ROOT histograms are 1-indexed
        }
      }
    }

    //--------------------------------------------------
    // functions for sanity check
    //--------------------------------------------------
    size_t EcondErrorSummarizer::getMatrixSize() const { return error_thresholds_matrix_.size(); }

    float EcondErrorSummarizer::getThreshold(const std::string& threshold_name) const {
      return config_data_[threshold_name];
    }

    std::string EcondErrorSummarizer::getErrorLabel(int error_type_index) const {
      if (error_type_index >= 0 && error_type_index < static_cast<int>(error_type_labels_.size())) {
        return error_type_labels_[error_type_index];
      }
      return "UNKNOWN";
    }

    void EcondErrorSummarizer::printConfigurationSummary() const {
      edm::LogPrint msg("HGCalDQMEcondErrorSummarizer");
      msg << "\n========== DQM Configuration Summary ==========\n";
      msg << "Total error types: " << error_thresholds_matrix_.size() << "\n";

      for (size_t i = 0; i < error_thresholds_matrix_.size(); ++i) {
        const auto& thresholds = error_thresholds_matrix_[i];
        msg << std::setw(2) << i << ". " << std::setw(25) << std::left << getErrorLabel(i) << " | Thresholds: [";

        for (size_t j = 0; j < thresholds.size(); ++j) {
          if (j > 0)
            msg << ", ";
          msg << thresholds[j];
        }
        msg << "]\n";
      }
      msg << "===============================================";
    }

    bool EcondErrorSummarizer::validateHistogramLabels(const MonitorElement* me) const {
      if (!me || me->kind() != MonitorElement::Kind::TH2F) {
        return false;
      }

      const TH2F* hist = const_cast<MonitorElement*>(me)->getTH2F();
      if (!hist)
        return false;

      bool all_match = true;
      for (int ybin = 1; ybin <= hist->GetNbinsY(); ++ybin) {
        std::string bin_label = hist->GetYaxis()->GetBinLabel(ybin);
        int expected_index = ybin - 1;
        std::string expected_label = getErrorLabel(expected_index);

        if (bin_label != expected_label) {
          edm::LogError("HGCalDQMEcondErrorSummarizer")
              << "MISMATCH at bin " << ybin << ": histogram='" << bin_label << "' vs config='" << expected_label << "'";
          all_match = false;
        }
      }

      return all_match;
    }

    //--------------------------------------------------
    // ECON-T Error Summarizer implementation
    //--------------------------------------------------
    EcontErrorSummarizer::EcontErrorSummarizer() : config_data_(json{}) {}

    EcontErrorSummarizer::EcontErrorSummarizer(const json& config_data) : config_data_(config_data) {
      loadMatrixConfiguration();
    }

    void EcontErrorSummarizer::loadMatrixConfiguration() {
      try {
        if (config_data_.contains("error_thresholds_matrix")) {
          auto matrix_json = config_data_["error_thresholds_matrix"];
          error_thresholds_matrix_.resize(matrix_json.size());

          for (size_t i = 0; i < matrix_json.size(); ++i) {
            error_thresholds_matrix_[i] = matrix_json[i].get<std::vector<int>>();
          }
        }

        if (config_data_.contains("error_type_labels")) {
          error_type_labels_ = config_data_["error_type_labels"].get<std::vector<std::string>>();
        }

        if (error_thresholds_matrix_.size() != error_type_labels_.size()) {
          edm::LogWarning("HGCalDQMEcontErrorSummarizer") << "Matrix size (" << error_thresholds_matrix_.size()
                                                          << ") != Labels size (" << error_type_labels_.size() << ")";
        }

      } catch (const std::exception& e) {
        edm::LogError("HGCalDQMEcontErrorSummarizer") << "Error loading matrix configuration: " << e.what();
      }
    }

    void EcontErrorSummarizer::processAndFill(const MonitorElement* source_me,
                                              MonitorElement* target_me,
                                              const int target_bin_id,
                                              const ProcessMode mode) {
      switch (mode) {
        case ProcessMode::STAT_TO_STAT:
          for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
            double total = 0;
            for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
              total += source_me->getBinContent(xbin, ybin);
            }
            if (total > 0) {
              target_me->setBinContent(target_bin_id + 1, ybin, total);
            }
          }
          break;

        case ProcessMode::STAT_TO_GRADE: {
          std::map<EconTErrorCategory, int> worst_grades = {{EconTErrorCategory::UNPACKING_ERRORS, 0},
                                                            {EconTErrorCategory::HEADER_TRAILER, 0}};

          for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
            int error_type_index = ybin - 1;
            if (!(error_type_index < static_cast<int>(EconTErrorType::NUM_ERROR_TYPES)))
              continue;
            EconTErrorCategory category = econdTErrorTypeToCategory[static_cast<EconTErrorType>(error_type_index)];

            for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
              double content = source_me->getBinContent(xbin, ybin);
              int current_grade = calculateGrade(error_type_index, static_cast<int>(content));
              worst_grades[category] = std::max(worst_grades[category], current_grade);
            }
          }

          for (size_t ybin = 1; ybin <= static_cast<size_t>(EconTErrorCategory::NUM_CATEGORIES); ++ybin) {
            EconTErrorCategory error_type_category = static_cast<EconTErrorCategory>(ybin - 1);
            target_me->setBinContent(target_bin_id + 1, ybin, worst_grades[error_type_category]);
          }
        } break;

        case ProcessMode::GRADE_TO_GRADE:
          for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
            int worst_grade = 0;
            for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
              int grade = source_me->getBinContent(xbin, ybin);
              worst_grade = std::max(worst_grade, grade);
            }
            target_me->setBinContent(target_bin_id + 1, ybin, worst_grade);
          }
          break;

        case ProcessMode::STAT_TO_GRADE_X: {
          std::vector<int> worst_grades;
          for (int xbin = 1; xbin <= source_me->getNbinsX(); ++xbin) {
            int worst_grade = 0;
            for (int ybin = 1; ybin <= source_me->getNbinsY(); ++ybin) {
              int error_type_index = ybin - 1;
              if (!(error_type_index < static_cast<int>(EconTErrorType::NUM_ERROR_TYPES)))
                continue;
              double content = source_me->getBinContent(xbin, ybin);
              int current_grade = calculateGrade(error_type_index, static_cast<int>(content));
              worst_grade = std::max(worst_grade, current_grade);
            }
            worst_grades.push_back(worst_grade);
          }

          for (size_t ybin = 1; ybin <= worst_grades.size(); ++ybin) {
            target_me->setBinContent(target_bin_id + 1, ybin, worst_grades[ybin - 1]);
          }
        } break;

        default:
          edm::LogError("HGCalDQMEcontErrorSummarizer") << "Invalid mode " << static_cast<int>(mode);
          break;
      }
    }

    int EcontErrorSummarizer::calculateGrade(int error_type_index, int num_errors) {
      if (error_type_index < 0 || error_type_index >= static_cast<int>(error_thresholds_matrix_.size())) {
        edm::LogError("HGCalDQMEcontErrorSummarizer") << "Invalid error type index " << error_type_index;
        return 5;
      }

      const auto& thresholds = error_thresholds_matrix_[error_type_index];
      for (size_t grade = 0; grade < thresholds.size(); ++grade) {
        if (num_errors <= thresholds[grade])
          return static_cast<int>(grade + 1);
      }

      return 5;
    }

    size_t EcontErrorSummarizer::getMatrixSize() const { return error_thresholds_matrix_.size(); }

    std::string EcontErrorSummarizer::getErrorLabel(int error_type_index) const {
      if (error_type_index >= 0 && error_type_index < static_cast<int>(error_type_labels_.size())) {
        return error_type_labels_[error_type_index];
      }
      return "UNKNOWN";
    }

    void EcontErrorSummarizer::printConfigurationSummary() const {
      edm::LogPrint msg("HGCalDQMEcontErrorSummarizer");
      msg << "\n========== ECON-T DQM Configuration Summary ==========\n";
      msg << "Total error types: " << error_thresholds_matrix_.size() << "\n";

      for (size_t i = 0; i < error_thresholds_matrix_.size(); ++i) {
        const auto& thresholds = error_thresholds_matrix_[i];
        msg << std::setw(2) << i << ". " << std::setw(25) << std::left << getErrorLabel(i) << " | Thresholds: [";

        for (size_t j = 0; j < thresholds.size(); ++j) {
          if (j > 0)
            msg << ", ";
          msg << thresholds[j];
        }
        msg << "]\n";
      }
      msg << "===============================================";
    }

    bool EcontErrorSummarizer::validateHistogramLabels(const MonitorElement* me) const {
      if (!me || me->kind() != MonitorElement::Kind::TH2F) {
        return false;
      }

      const TH2F* hist = const_cast<MonitorElement*>(me)->getTH2F();
      if (!hist)
        return false;

      bool all_match = true;
      for (int ybin = 1; ybin <= hist->GetNbinsY(); ++ybin) {
        std::string bin_label = hist->GetYaxis()->GetBinLabel(ybin);
        int expected_index = ybin - 1;
        std::string expected_label = getErrorLabel(expected_index);

        if (bin_label != expected_label) {
          edm::LogError("HGCalDQMEcontErrorSummarizer")
              << "MISMATCH at bin " << ybin << ": histogram='" << bin_label << "' vs config='" << expected_label << "'";
          all_match = false;
        }
      }

      return all_match;
    }

    std::vector<HGCalDQMModule> readoutModules(HGCalMappingModuleIndexer const& moduleIndexer,
                                               hgcal::HGCalMappingModuleParamHost const& moduleInfo) {
      std::vector<HGCalDQMModule> modules;
      modules.reserve(moduleIndexer.typecodeMap().size());
      for (auto const& [rawTypecode, fedData] : moduleIndexer.typecodeMap()) {
        auto const [fedid, imod] = fedData;
        uint32_t const denseModIdx = moduleIndexer.getIndexForModule(fedid, imod);
        auto const& modInfo = moduleInfo.view()[denseModIdx];

        HGCalDQMModule ele;
        ele.typecode = rawTypecode;
        std::replace(ele.typecode.begin(), ele.typecode.end(), '-', '_');
        ele.dqmIndex = denseModIdx;
        ele.nErx = moduleIndexer.getNumERxs(fedid, imod);
        ele.zside = modInfo.zside();
        ele.endcap = ele.zside ? 1 : -1;
        ele.isSiPM = modInfo.isSiPM();
        ele.layer = modInfo.plane();
        ele.i1 = modInfo.i1();
        ele.i2 = modInfo.i2();
        ele.fedid = fedid;
        ele.modid = imod;
        ele.econdidx = modInfo.econdidx();
        ele.cassette = modInfo.cassette();
        modules.push_back(std::move(ele));
      }
      return modules;
    }

  }  // namespace dqm
}  // namespace hgcal
