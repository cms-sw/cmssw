#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloLayer1Tokens.h"
#include "CaloLayer1Packer.h"

namespace l1t {
namespace stage2 {
   Blocks
   CaloLayer1Packer::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EcalTrigPrimDigiCollection> ecalDigis;
      event.getByToken(static_cast<const CaloLayer1Tokens*>(toks)->getEcalDigiToken(), ecalDigis);
      edm::Handle<HcalTrigPrimDigiCollection> hcalDigis;
      event.getByToken(static_cast<const CaloLayer1Tokens*>(toks)->getHcalDigiToken(), hcalDigis);
      edm::Handle<L1CaloRegionCollection> caloRegions;
      event.getByToken(static_cast<const CaloLayer1Tokens*>(toks)->getCaloRegionToken(), caloRegions);

      auto ctp7_phi = board();

      std::vector<uint32_t> load;
      load.resize(192);

      unsigned bx_per_l1a = 1;
      // CTP7 uses CMS scheme, starting at 0
      // TODO: expected +2, but +1 apparently?
      unsigned calo_bxid = (event.bunchCrossing()+1) % 3564;

      // a la CTP7Payload::getHeader()
      unsigned blockId = 0;
      unsigned blockSize = 192;
      unsigned capId = 0;
      unsigned blockFlags = ((bx_per_l1a&0xf)<<16) | (calo_bxid&0xfff);
      BlockHeader hdr(blockId, blockSize, capId, blockFlags, CTP7);
      Block block(hdr, &*load.begin(), &*load.end());

      Blocks res;
      res.push_back(block);
      return res;
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::CaloLayer1Packer);
