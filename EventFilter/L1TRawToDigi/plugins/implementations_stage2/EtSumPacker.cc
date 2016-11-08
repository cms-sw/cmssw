#include "FWCore/Framework/interface/Event.h"
<<<<<<< HEAD:EventFilter/L1TRawToDigi/plugins/implementations_stage2/EtSumPacker.cc
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
=======
>>>>>>> cms-sw/refs/pull/15748/head:EventFilter/L1TRawToDigi/src/implementations_stage2/EtSumPacker.cc

#include "CaloTokens.h"

#include "L1TStage2Layer2Constants.h"
#include "EtSumPacker.h"

namespace l1t {
namespace stage2 {
   Blocks
   EtSumPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EtSumBxCollection> etSums;
      event.getByToken(static_cast<const CommonTokens*>(toks)->getEtSumToken(), etSums);

      uint32_t et_word = 0;
      uint32_t ht_word = 0;
      uint32_t met_word = 0;
      uint32_t mht_word = 0;

      for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
         for (auto j = etSums->begin(i); j != etSums->end(i); ++j) {
	   uint32_t word = std::min(j->hwPt(), 0xFFF);
	   if ((j->getType()==l1t::EtSum::kMissingEt) || (j->getType()==l1t::EtSum::kMissingHt))
	     word = word | ((j->hwPhi() & 0xFF) << 12);
	   
	   if (j->getType()==l1t::EtSum::kTotalEt)   et_word = word;
	   if (j->getType()==l1t::EtSum::kTotalHt)   ht_word = word;
	   if (j->getType()==l1t::EtSum::kMissingEt) met_word = word;
	   if (j->getType()==l1t::EtSum::kMissingHt) mht_word = word;
         }
      }

      std::vector<uint32_t> load;
      load.push_back(et_word);
      load.push_back(ht_word);
      load.push_back(met_word);
      load.push_back(mht_word);
      while (load.size()<l1t::stage2::layer2::demux::nOutputFramePerBX) load.push_back(0);

      return {Block(b1_, load)};
   }
}
}

<<<<<<< HEAD:EventFilter/L1TRawToDigi/plugins/implementations_stage2/EtSumPacker.cc
DEFINE_L1T_PACKER(l1t::stage2::CaloEtSumPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTEtSumPacker);
=======
// moved to plugins/SealModule.cc
// DEFINE_L1T_PACKER(l1t::stage2::CaloEtSumPacker);
// DEFINE_L1T_PACKER(l1t::stage2::GTEtSumPacker);

>>>>>>> cms-sw/refs/pull/15748/head:EventFilter/L1TRawToDigi/src/implementations_stage2/EtSumPacker.cc
