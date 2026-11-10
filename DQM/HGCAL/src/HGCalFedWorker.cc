#include "DQM/HGCAL/interface/HGCalFedWorker.h"

#include "DQM/HGCAL/interface/HGCalDQMGeometry.h"

namespace hgcal {
  namespace dqm {

    HGCalFedWorker::HGCalFedWorker(std::string folderRoot) : folderRoot_(std::move(folderRoot)) {}

    void HGCalFedWorker::book(DQMStore::IBooker& ibooker, HGCalDQMGeometry const& /*geom*/) {
      ibooker.setCurrentFolder(folderRoot_ + "/FED");
      // Binning must match the payload axis of fedPayload booked in HGCalFastStreamDQM.
      me_fed_payload_th1d_ =
          ibooker.book1D("fed_payload_distribution", "FED Payload Distribution;Payload;Counts", 100, 0, 4000);
    }

    void HGCalFedWorker::endLumi(DQMStore::IBooker& /*ibooker*/,
                                 DQMStore::IGetter& igetter,
                                 HGCalDQMGeometry& /*geom*/,
                                 edm::LuminosityBlock const& /*iLumi*/) {
      // project all FED payload
      MonitorElement* me = igetter.get(folderRoot_ + "/FED/fedPayload");

      if (!me || !(me->getTH2F())) {
        edm::LogWarning("HGCalFedWorker") << "Could not find fedPayload histogram";
        return;
      }

      // 1D payload distribution using TH2F::ProjectionY()
      // fedPayload accumulates over the run, so replace rather than add to the previous projection.
      TH1* me_proj = me->getTH2F()->ProjectionY();
      me_fed_payload_th1d_->Reset();
      me_fed_payload_th1d_->getTH1()->Add(me_proj);
      delete me_proj;
    }

  }  // namespace dqm
}  // namespace hgcal
