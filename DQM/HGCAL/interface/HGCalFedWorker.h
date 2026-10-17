#ifndef DQM_HGCAL_interface_HGCalFedWorker_h
#define DQM_HGCAL_interface_HGCalFedWorker_h

#include <string>

#include "DQM/HGCAL/interface/HGCalDQMWorkerBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace hgcal {
  namespace dqm {

    using MonitorElement = ::dqm::impl::MonitorElement;

    // Books HGCAL/FED/fed_payload_distribution and refills it every lumisection with the
    // Y projection of HGCAL/FED/fedPayload. FED-scoped quality MEs (econdQualityFED_*,
    // econtQualityFED_*) are client MEs not touched here.
    class HGCalFedWorker : public HGCalDQMWorkerBase {
    public:
      explicit HGCalFedWorker(std::string folderRoot);
      ~HGCalFedWorker() override = default;

      void book(DQMStore::IBooker&, HGCalDQMGeometry const&) override;

      void endLumi(DQMStore::IBooker&, DQMStore::IGetter&, HGCalDQMGeometry&, edm::LuminosityBlock const&) override;

    private:
      std::string folderRoot_;
      MonitorElement* me_fed_payload_th1d_ = nullptr;
    };

  }  // namespace dqm
}  // namespace hgcal

#endif
