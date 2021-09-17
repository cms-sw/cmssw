#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include "BMTFTokens.h"

namespace l1t {
  namespace stage2 {
    class BMTFPackerOutput : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
      void setKalmanAlgoTrue() { isKalman_ = true; };

    private:
      std::map<unsigned int, std::vector<uint32_t> > payloadMap_;

      bool isKalman_{false};
    };
  }  // namespace stage2
}  // namespace l1t
