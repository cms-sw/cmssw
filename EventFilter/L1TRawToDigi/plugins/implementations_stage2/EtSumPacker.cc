#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

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

      std::vector<uint32_t> load;
      int nBx = 0;
      
      for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
	
	uint32_t et_word = 0;
	uint32_t ht_word = 0;
	uint32_t met_word = 0;
	uint32_t mht_word = 0;
	uint32_t methf_word = 0;
	uint32_t mhthf_word = 0;
	
	for (auto j = etSums->begin(i); j != etSums->end(i); ++j) {
	  uint32_t word = std::min(j->hwPt(), 0xFFF);
	  if ((j->getType()==l1t::EtSum::kMissingEt) || (j->getType()==l1t::EtSum::kMissingHt) 
	      || (j->getType()==l1t::EtSum::kMissingEtHF) || (j->getType()==l1t::EtSum::kMissingHtHF))
	    word = word | ((j->hwPhi() & 0xFF) << 12);
	  
	  if (j->getType()==l1t::EtSum::kTotalEt)       et_word  |= word;
	  if (j->getType()==l1t::EtSum::kTotalEtEm)     et_word  |= (word << 12);
	  if (j->getType()==l1t::EtSum::kMinBiasHFP0)   et_word  |= (word << 28);
	  if (j->getType()==l1t::EtSum::kTotalHt)       ht_word  |= word;
	  if (j->getType()==l1t::EtSum::kMinBiasHFM0)   ht_word  |= (word << 28);
	  if (j->getType()==l1t::EtSum::kMissingEt)     met_word |= word;
	  if (j->getType()==l1t::EtSum::kMinBiasHFP1)   met_word |= (word << 28);
	  if (j->getType()==l1t::EtSum::kMissingHt)     mht_word |= word;
	  if (j->getType()==l1t::EtSum::kMinBiasHFM1)   mht_word |= (word << 28);
	  if (j->getType()==l1t::EtSum::kMissingEtHF)   methf_word |= word;
	  if (j->getType()==l1t::EtSum::kMissingHtHF)   mhthf_word |= word;
	  if (j->getType()==l1t::EtSum::kTowerCount)    ht_word |= (word << 12);
	  if (j->getType()==l1t::EtSum::kAsymEt)        met_word |= (word << 20);
	  if (j->getType()==l1t::EtSum::kAsymHt)        mht_word |= (word << 20);
	  if (j->getType()==l1t::EtSum::kAsymEtHF)      methf_word |= (word << 20);
	  if (j->getType()==l1t::EtSum::kAsymHtHF)      mhthf_word |= (word << 20);
	  if (j->getType()==l1t::EtSum::kCentrality){
	    methf_word |= ((word & 0xF) << 28);
	    mhthf_word |= (((word>>4) & 0xF) << 28);
	  } 
	  
	}
	
	load.push_back(et_word);
	load.push_back(ht_word);
	load.push_back(met_word);
	load.push_back(mht_word);
	load.push_back(methf_word);
	load.push_back(mhthf_word);

	//pad with zeros to fill out block; must do this for each BX
	while (load.size() -nBx*l1t::stage2::layer2::demux::nOutputFramePerBX <l1t::stage2::layer2::demux::nOutputFramePerBX) load.push_back(0);
	nBx++;
      
      }
      

      return {Block(b1_, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::CaloEtSumPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTEtSumPacker);
