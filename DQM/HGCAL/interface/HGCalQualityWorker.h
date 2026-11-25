#ifndef DQM_HGCAL_interface_HGCalQualityWorker_h
#define DQM_HGCAL_interface_HGCalQualityWorker_h

#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#include "DQM/HGCAL/interface/HGCalDQMWorkerBase.h"

namespace hgcal {
  namespace dqm {

    using MonitorElement = ::dqm::impl::MonitorElement;

    class EcondErrorSummarizer;  // injected; plugin owns

    // Books/fills ECON-D quality + payload summaries, updated every lumisection.
    // The ECON-T equivalent lives in HGCalTriggerWorker.
    class HGCalQualityWorker : public HGCalDQMWorkerBase {
    public:
      HGCalQualityWorker(std::string folderRoot, EcondErrorSummarizer& econdErrorSummarizer);
      ~HGCalQualityWorker() override = default;

      void book(DQMStore::IBooker&, HGCalDQMGeometry const&) override;
      void endLumi(DQMStore::IBooker&, DQMStore::IGetter&, HGCalDQMGeometry&, edm::LuminosityBlock const&) override;

    private:
      std::string folderRoot_;
      EcondErrorSummarizer& econd_error_summarizer_;  // NOT owned; plugin owns
      bool first_run_ = true;                         // label validation runs once per worker

      int findBinByLabel(MonitorElement* me, const std::string& label) const;

      mutable std::unordered_map<MonitorElement*, std::unordered_map<std::string, int>> label_to_bin_cache_;

      MonitorElement* me_econd_quality_summary_ = nullptr;
      std::map<int, std::map<int, MonitorElement*>> econdQualityLayer_;
      std::map<int, std::map<int, MonitorElement*>> econdPayloadLayer_;

      // Fast-stream ECON-D hex plots: plotKey -> layer -> TH2Poly.
      // Keys: "econdQuality", "avgPayload", "stdPayload".
      std::map<std::string, std::map<int, MonitorElement*>> hexPlotsFastStream_;

      const std::vector<std::string> hexPlotsFastStreamKey_ = {"econdQuality", "avgPayload", "stdPayload"};
      std::map<int, std::string> endCapKey_ = {{-1, "Minus"}, {1, "Plus"}};
    };

  }  // namespace dqm
}  // namespace hgcal

#endif
