#ifndef DQM_HGCAL_interface_HGCalChannelWorker_h
#define DQM_HGCAL_interface_HGCalChannelWorker_h

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "DQM/HGCAL/interface/HGCalDQMWorkerBase.h"

namespace hgcal {
  namespace dqm {

    using MonitorElement = ::dqm::impl::MonitorElement;

    class EcondErrorSummarizer;  // injected; plugin owns

    // Books/fills per-module channel-level TH2Poly plots, per-cassette +
    // per-layer TH2Poly summaries over ~13 variables, and per-FED
    // summaryPerModule under HGCAL/FED/FED_<id>/.
    class HGCalChannelWorker : public HGCalDQMWorkerBase {
    public:
      HGCalChannelWorker(std::string folderRoot,
                         EcondErrorSummarizer& econdErrorSummarizer,
                         float overflowThreshold,
                         float saturatedAdcThreshold,
                         bool enableOverflowMarkers);
      ~HGCalChannelWorker() override = default;

      void book(DQMStore::IBooker&, HGCalDQMGeometry const&) override;
      void endRun(DQMStore::IBooker&, DQMStore::IGetter&, HGCalDQMGeometry&) override;

    private:
      std::string folderRoot_;
      EcondErrorSummarizer& econd_error_summarizer_;  // NOT owned; plugin owns

      float overflow_threshold_;
      float saturated_adc_threshold_;
      bool enable_overflow_markers_;

      // Index into variables_ used in the per-channel loop; the order must match variables_.
      enum VarIdx {
        enumIDX_occupancy = 0,          // "occupancy"
        enumIDX_avgcm = 1,              // "avgcm"
        enumIDX_avgadc = 2,             // "avgadc"
        enumIDX_stdadc = 3,             // "stdadc"
        enumIDX_deltaadc = 4,           // "deltaadc"
        enumIDX_avgtoa = 5,             // "avgtoa"
        enumIDX_avgtot = 6,             // "avgtot"
        enumIDX_n_vacant_channels = 7,  // "n_vacant_channels"
        enumIDX_avgmips = 8,            // "avgmips"
        enumIDX_stdmips = 9,            // "stdmips"
        enumIDX_toaoccupancy = 10,      // "toaoccupancy"
        enumIDX_totoccupancy = 11,      // "totoccupancy"
        N_VARS = 12                     // total count - always keep this last
      };

      // Layer -> variable -> whole-layer TH2Poly, booked in book() and filled in endRun().
      // Kept as a map: also holds "noisy"/"stuck"/"saturated" keys outside VarIdx.
      std::map<int, std::map<std::string, MonitorElement*>> hexLayer_;

      // Layer -> cassette -> VarIdx -> TH2Poly, booked in book() and filled in endRun().
      std::map<int, std::map<int, std::array<MonitorElement*, N_VARS>>> hexCassette_;

      // dqmIndex -> VarIdx -> module-level TH2Poly (channel-granularity).
      std::map<uint32_t, std::array<MonitorElement*, N_VARS>> hexPlots_;

      // Per-layer 1D summaries over the base variable set.
      std::map<std::string, MonitorElement*> Layer_;

      // Per-module stdadc TProfile, keyed by dqmIndex.
      std::map<uint32_t, MonitorElement*> stdadc_me_;

      // Per-FED summaryPerModule_FED<fedid> histogram.
      std::map<uint32_t, MonitorElement*> summary_ME_perFED_;

      // Base variable set; layer plots extend this with noisy/stuck/saturated
      // (added in the per-layer booking loop). SummaryLabel map lives in the .cc.
      const std::vector<std::string> variables_ = {"occupancy",
                                                   "avgcm",
                                                   "avgadc",
                                                   "stdadc",
                                                   "deltaadc",
                                                   "avgtoa",
                                                   "avgtot",
                                                   "n_vacant_channels",
                                                   "avgmips",
                                                   "stdmips",
                                                   "toaoccupancy",
                                                   "totoccupancy"};

      std::map<int, std::string> endCapKey_ = {{-1, "Minus"}, {1, "Plus"}};
    };

  }  // namespace dqm
}  // namespace hgcal

#endif
