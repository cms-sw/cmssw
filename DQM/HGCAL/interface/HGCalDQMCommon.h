#ifndef DQM_HGCAL_interface_HGCalDQMCommon_h
#define DQM_HGCAL_interface_HGCalDQMCommon_h

#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
// The SoA headers pulled in below need Eigen/Core first, or later SoA layouts with Eigen columns fail to compile.
#include <Eigen/Core>
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using json = nlohmann::json;
using dqm::impl::MonitorElement;

namespace hgcal {

  namespace dqm {

    // @short an enum for the final quantities displayed on hexplots
    enum SummaryIndices_t {
      CMAVG = 0,
      PEDESTAL,
      NOISE,
      DELTAPEDESTAL,
      TOAAVG,
      TOTAVG,
      NMIPSAVG,
      NMIPSSTD,
      LASTSUMMARYINDEX
    };

    // @short label for SummaryIndices_t enum (ROOT format)
    std::string getLabelForSummaryIndex(SummaryIndices_t idx);

    // common helper functions
    void addBinLabels(std::vector<std::string>& binlabels, MonitorElement* hist, int binPos);

    // Define an enum for ECON-D error types
    enum class EcondErrorType {
      // ECOND flags
      HT_GOOD = 0,
      HT_FAIL,
      HT_AMB,
      EBO_GOOD,
      EBO_FAIL,
      EBO_AMB,
      UNMATCHED,
      TRUNC,
      UNEXPECTED,
      SUB_PACKET_ERROR,
      MARKER,
      PAYLOAD_OF,
      PAYLOAD_MISMATCH,

      // CB flags
      CB_NOT_NORMAL,
      CB_PAYLOAD,
      CRC_ERROR,
      EVID_MISMATCH,
      FSM_TIMEOUT,
      BCID_ORBITID,
      MB_OVERFLOW,
      INACTIVE,
      CRC_TRAILER_ERROR,

      NUM_ERROR_TYPES
    };

    // Define an enum for quality categories
    enum class ErrorCategory {
      CB_ISSUES = 0,       // Capture block associated problems
      ECOND_PAYLOAD = 1,   // Endcap concentrator data packet problems
      HEADER_TRAILER = 2,  // Basic data packet format problems
      NUM_CATEGORIES = 3
    };

    // Define process mode for EcondErrorSummarizer::processAndFill() method
    enum class ProcessMode { STAT_TO_STAT = 0, STAT_TO_GRADE = 1, GRADE_TO_GRADE = 2, STAT_TO_GRADE_X = 3 };

    enum class EconTErrorType { NTC_NOT_MATCHING = 0, SUBPACKET_ERROR = 1, TDAQIDX_OUT_RANGE = 2, NUM_ERROR_TYPES = 3 };

    enum class EconTErrorCategory { UNPACKING_ERRORS = 0, HEADER_TRAILER = 1, NUM_CATEGORIES = 2 };

    // ECON-D / T quality flags & Helper functions to convert between enum and string
    extern std::vector<std::string> econdWithCBflags;
    extern std::vector<std::string> qualityCategoryNames, econtQualityCategoryNames;
    extern std::map<EcondErrorType, ErrorCategory> errorTypeToCategory;

    /**
       @short converts an error type to a human-readable string
     */
    extern std::string econdErrorTypeToString(EcondErrorType type);

    /**
       @short returns the sequential bin labels for the ECON-D/CB flags histograms
    */
    extern EcondErrorType stringToEcondErrorType(const std::string& errorString);

    /**
       @short decodes the flags and returns the list of bins to be filled
    */
    extern std::vector<int> getErrorBinsForECONDCBFlags(unsigned int cbflag,
                                                        unsigned int cbflags,
                                                        unsigned int exception);

    extern std::vector<std::string> econdTFlags;
    extern std::map<EconTErrorType, EconTErrorCategory> econdTErrorTypeToCategory;
    extern std::string econdTErrorTypeToString(EconTErrorType type);
    extern EconTErrorType stringToEconTErrorType(const std::string& errorString);

    // Summarize error from a provided monitor element for ECON-T
    class EcontErrorSummarizer {
    public:
      EcontErrorSummarizer();
      EcontErrorSummarizer(const json& config_data);
      ~EcontErrorSummarizer() = default;

      // accumulate stats from source and fill the total stat into a specified bin of target monitor element
      void processAndFill(const MonitorElement* source_me,
                          MonitorElement* target_me,
                          const int target_bin_id,
                          const ProcessMode mode = ProcessMode::STAT_TO_STAT);

      // calculate quality grade (1=best, 5=worst) based on error count for given flag
      int calculateGrade(int error_type_index, int num_errors);

      // getters for sanity checks
      std::string getErrorLabel(int error_type_index) const;
      size_t getMatrixSize() const;
      void printConfigurationSummary() const;
      bool validateHistogramLabels(const MonitorElement* me) const;

    private:
      json config_data_;

      void loadMatrixConfiguration();
      std::vector<std::vector<int>> error_thresholds_matrix_;
      std::vector<std::string> error_type_labels_;
    };

    using CategoryID = uint32_t;

    struct Report {
      // Use an array indexed by the enum
      std::map<CategoryID, std::array<int, static_cast<size_t>(EcondErrorType::NUM_ERROR_TYPES)>> categorized_counters;

      // Constructor initializes the array to zeros
      Report() : categorized_counters{} {}

      // Utility methods
      void addError(EcondErrorType type, int count, CategoryID category);
      int getCategoryErrorCount(CategoryID category, EcondErrorType type) const;
      int getCategoryTotalErrors(CategoryID category) const;
      std::string getCategoryErrorSummary(CategoryID category) const;
      std::vector<CategoryID> getAllCategories() const;
    };

    struct BoundingBox {
      float xmin, xmax, ymin, ymax;

      BoundingBox()
          : xmin(std::numeric_limits<float>::max()),
            xmax(std::numeric_limits<float>::lowest()),
            ymin(std::numeric_limits<float>::max()),
            ymax(std::numeric_limits<float>::lowest()) {}

      BoundingBox(float xmin_, float xmax_, float ymin_, float ymax_)
          : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_) {}

      bool isValid() const { return xmin <= xmax && ymin <= ymax; }
    };

    // Summarize error from a provided monitor element
    class EcondErrorSummarizer {
    public:
      EcondErrorSummarizer();
      EcondErrorSummarizer(const json& config_data);
      ~EcondErrorSummarizer() = default;

      // accumulate stats from srouce and fill the total stat into a specified bin of target monitor element
      void processAndFill(const MonitorElement* source_me,
                          MonitorElement* target_me,
                          const int target_bin_id,
                          const ProcessMode mode = ProcessMode::STAT_TO_STAT);

      // calls processAndFills three times for the three different versus LS plots.
      void processAndFillLS(MonitorElement* source_quality_layer,
                            int binNumber,
                            MonitorElement* target_quality_LS,
                            MonitorElement* target_finequality_LS,
                            MonitorElement* target_layer_LS);

      // calculate quality grade (1=best, 5=worst) based on error count for given flag
      int calculateGrade(int error_type_index, int num_errors);

      // analyze the numbers of stuck/noisy/normal channels of a module
      std::map<std::string, int> count_zero_std_bins(const TProfile* profile, int nMax);
      static std::map<std::string, int> analyzeChannelQuality(MonitorElement* me, double threshold);
      std::map<std::string, int> analyzeChannelQuality(MonitorElement* me, const std::string& threshold_key);
      std::vector<double> sumAxis(MonitorElement* me, int axis);

      // getters for sanity checks
      float getThreshold(const std::string& threshold_name) const;
      std::string getErrorLabel(int error_type_index) const;
      size_t getMatrixSize() const;
      void printConfigurationSummary() const;
      bool validateHistogramLabels(const MonitorElement* me) const;

      // preduce report for higher level TH2F, i.e. modules per cassette -> cassettes per layer -> layers per endcap
      Report analysisWithCategories(MonitorElement* me, const std::function<CategoryID(int)>& categorizer);
      static Report aggregateReports(const std::vector<Report>& reports);
      void fillHistogramFromCategoryReport(MonitorElement* me, const Report& report, CategoryID category);

    private:
      json config_data_;

      void loadMatrixConfiguration();
      std::vector<std::vector<int>> error_thresholds_matrix_;
      std::vector<std::string> error_type_labels_;
    };

    // A DAQ module of the electronics mapping, as used by the DQM clients.
    // typecode has '-' replaced by '_'; endcap is +1/-1 stored as uint32_t, as the clients use it.
    // moduleIndex and fedModuleIndex are left for the caller to fill.
    struct HGCalDQMModule {
      std::string typecode;
      bool zside{false}, isSiPM{false};
      uint32_t layer{0}, i1{0}, i2{0}, nErx{0}, dqmIndex{0}, fedid{0}, modid{0}, econdidx{0}, cassette{0}, endcap{0},
          moduleIndex{0}, fedModuleIndex{0};
    };

    // All modules of the mapping, in moduleIndexer.typecodeMap() order.
    std::vector<HGCalDQMModule> readoutModules(HGCalMappingModuleIndexer const& moduleIndexer,
                                               hgcal::HGCalMappingModuleParamHost const& moduleInfo);

  }  // namespace dqm

}  // namespace hgcal

#endif
