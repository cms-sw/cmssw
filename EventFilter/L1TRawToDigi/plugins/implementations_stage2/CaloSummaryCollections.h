#ifndef EventFilter_L1TRawToDigi_CaloSummaryCollections_h
#define EventFilter_L1TRawToDigi_CaloSummaryCollections_h

#include "DataFormats/L1CaloTrigger/interface/CICADA.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
  namespace stage2 {
    class CaloSummaryCollections : public UnpackerCollections {
    public:
      CaloSummaryCollections(edm::Event& e)
          : UnpackerCollections(e), cicadaDigis_(std::make_unique<CICADABxCollection>()){};
      ~CaloSummaryCollections() override;
      inline CICADABxCollection* getCICADABxCollection() { return cicadaDigis_.get(); };

    private:
      std::unique_ptr<CICADABxCollection> cicadaDigis_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif
