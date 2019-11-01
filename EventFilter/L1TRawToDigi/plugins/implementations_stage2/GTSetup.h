#ifndef L1T_PACKER_STAGE2_GTSETUP_H
#define L1T_PACKER_STAGE2_GTSETUP_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "GTCollections.h"
#include "GTTokens.h"

namespace l1t {
  namespace stage2 {
    class GTSetup : public PackingSetup {
    public:
      ~GTSetup() override{};
      std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override;
      void fillDescription(edm::ParameterSetDescription& desc) override;
      PackerMap getPackers(int fed, unsigned int fw) override;
      void registerProducts(edm::ProducesCollector) override;
      std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override;
      UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override;
    };
  }  // namespace stage2
}  // namespace l1t

#endif
