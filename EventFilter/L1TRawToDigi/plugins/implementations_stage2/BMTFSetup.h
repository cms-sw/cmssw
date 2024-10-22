#ifndef L1T_PACKER_STAGE2_BMTFSETUP_H
#define L1T_PACKER_STAGE2_BMTFSETUP_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "BMTFCollections.h"
#include "BMTFTokens.h"
#include "BMTFUnpackerOutput.h"
#include "BMTFPackerOutput.h"

namespace l1t {
  namespace stage2 {
    class BMTFSetup : public PackingSetup {
    public:
      std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override;
      void fillDescription(edm::ParameterSetDescription& desc) override;
      PackerMap getPackers(int fed, unsigned int fw) override;
      void registerProducts(edm::ProducesCollector) override;
      std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override;
      UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override;

    private:
      const std::map<int, int> boardIdPerSlot{
          // {slot, boardId}
          {1, 1},
          {3, 2},
          {5, 3},
          {7, 4},
          {9, 5},
          {11, 6},  // Top Crate
          {2, 7},
          {4, 8},
          {6, 9},
          {8, 10},
          {10, 11},
          {12, 12}  // Bottom Crate
      };
      const unsigned int firstNewInputsFwVer = 0x92300120;
      const unsigned int firstKalmanFwVer = 0x95000160;
    };
  }  // namespace stage2
}  // namespace l1t

#endif
