#ifndef DQM_HGCAL_interface_HGCalTriggerWorker_h
#define DQM_HGCAL_interface_HGCalTriggerWorker_h

#include <map>
#include <string>
#include <vector>

#include "DQM/HGCAL/interface/HGCalDQMWorkerBase.h"

namespace hgcal {
  namespace dqm {

    // Same MonitorElement type as HGCalDQMCommon.h; another one would not link with EcontErrorSummarizer.
    using MonitorElement = ::dqm::impl::MonitorElement;

    class EcontErrorSummarizer;  // injected by plugin; forward-declared to keep
                                 // json.hpp out of this header

    // book() books per-BX hexagonal TH2Poly plots (layer/cassette/module scope)
    // plus the stage-1 summary and ECON-T quality surfaces.
    // endRun() fills all of it from client MEs at "HGCAL/Trigger/..." into folderRoot_/Trigger.
    class HGCalTriggerWorker : public HGCalDQMWorkerBase {
    public:
      // econtErrorSummarizer must outlive the worker (plugin-owned).
      HGCalTriggerWorker(std::string folderRoot, EcontErrorSummarizer& econtErrorSummarizer);
      ~HGCalTriggerWorker() override = default;

      void book(DQMStore::IBooker&, HGCalDQMGeometry const&) override;
      void endRun(DQMStore::IBooker&, DQMStore::IGetter&, HGCalDQMGeometry&) override;

    private:
      void fillHexaPlots(DQMStore::IGetter&, HGCalDQMGeometry&);
      void fillStage1Summary(DQMStore::IGetter&, HGCalDQMGeometry const&);
      void fillEcontQuality(DQMStore::IGetter&, HGCalDQMGeometry const&);

      std::string folderRoot_;
      EcontErrorSummarizer& econt_error_summarizer_;  // NOT owned; plugin owns

      std::map<int, std::map<std::string, MonitorElement*>> hexTriggerLayer_;
      std::map<int, std::map<int, std::map<std::string, MonitorElement*>>> hexTriggerCassette_;
      std::map<std::string, std::map<std::string, MonitorElement*>> hexTriggerPlots_;
      MonitorElement* me_econt_stage1_summary_ = nullptr;

      // ECON-T quality aggregation, fed from TPGDQM's econtQualityCassette_<c>:
      //   econtQuality (layer x category)                 under Trigger/
      //   econtQualityLayer_<abs(layer)> (cassette x cat) under Trigger/Endcap_X/Layer_N/
      MonitorElement* me_econt_quality_summary_ = nullptr;
      std::map<int, std::map<int, MonitorElement*>> econtQualityLayer_;
      std::map<int, std::string> endCapKey_ = {{-1, "Minus"}, {1, "Plus"}};

      const std::vector<std::string> BXlist_ = {"BXm3", "BXm2", "BXm1", "BX0", "BXp1", "BXp2", "BXp3"};
    };

  }  // namespace dqm
}  // namespace hgcal

#endif
