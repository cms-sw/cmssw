#ifndef DQM_HGCAL_interface_HGCalLSWorker_h
#define DQM_HGCAL_interface_HGCalLSWorker_h

#include <string>

#include "DQM/HGCAL/interface/HGCalDQMWorkerBase.h"

namespace edm {
  class LuminosityBlock;
}

namespace hgcal {
  namespace dqm {

    using MonitorElement = ::dqm::impl::MonitorElement;

    class EcondErrorSummarizer;
    class EcontErrorSummarizer;

    // Per-LS quality trend plots using SetRangeUser approach.
    // Books histograms once with kMaxLS bins and zooms the x-axis
    // to the filled range after each fill.
    class HGCalLSWorker : public HGCalDQMWorkerBase {
    public:
      HGCalLSWorker(std::string folderRoot,
                    EcondErrorSummarizer& econdErrorSummarizer,
                    EcontErrorSummarizer& econtErrorSummarizer,
                    bool skipTriggerDQM);
      ~HGCalLSWorker() override = default;

      void book(DQMStore::IBooker&, HGCalDQMGeometry const&) override;

      void endLumi(DQMStore::IBooker&, DQMStore::IGetter&, HGCalDQMGeometry&, edm::LuminosityBlock const&) override;

    private:
      std::string folderRoot_;
      EcondErrorSummarizer& econd_error_summarizer_;
      EcontErrorSummarizer& econt_error_summarizer_;
      bool skipTriggerDQM_;

      static constexpr int kMaxLS = 2000;

      // ECON-D per-LS MEs
      MonitorElement* me_econd_quality_LS_ = nullptr;
      MonitorElement* me_econd_finequality_LS_ = nullptr;
      MonitorElement* me_econd_layer_LS_ = nullptr;
      MonitorElement* me_econd_quality_layer_ = nullptr;

      // ECON-T per-LS MEs (booked only if !skipTriggerDQM_)
      MonitorElement* me_econt_quality_LS_ = nullptr;
      MonitorElement* me_econt_finequality_LS_ = nullptr;
      MonitorElement* me_econt_layer_LS_ = nullptr;
      MonitorElement* me_econt_quality_layer_ = nullptr;
    };

  }  // namespace dqm
}  // namespace hgcal

#endif
