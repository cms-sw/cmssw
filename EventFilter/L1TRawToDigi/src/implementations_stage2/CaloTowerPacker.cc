#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage2 {
      class CaloTowerPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   CaloTowerPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {

      edm::Handle<CaloTowerBxCollection> towers;
      event.getByToken(static_cast<const CaloTokens*>(toks)->getCaloTowerToken(), towers);

      Blocks res;

      for (int i = towers->getFirstBX(); i <= towers->getLastBX(); ++i) {

        for (int phi = 1; phi <=72; phi=phi+2) { // Two phi values per link

          unsigned int id = 2*phi - 2; // Block IDs start at zero and span even numbers up to 142
          std::vector<uint32_t> load;

          for (int eta = 1; eta <=41; eta++) { // This is abs(eta) since +/- eta are interleaved in time

	    if (eta==CaloTools::kHFBegin) continue;

            // Get four towers +/- eta and phi and phi+1 to all be packed in this loop
            l1t::CaloTower t1 = towers->at(i,l1t::CaloTools::caloTowerHash(eta,phi));
            l1t::CaloTower t2 = towers->at(i,l1t::CaloTools::caloTowerHash(eta,phi+1));
            l1t::CaloTower t3 = towers->at(i,l1t::CaloTools::caloTowerHash(-1*eta,phi));
            l1t::CaloTower t4 = towers->at(i,l1t::CaloTools::caloTowerHash(-1*eta,phi+1));

            // Merge phi and phi+1 into one block (phi is LSW, phi+1 is MSW)
            uint32_t word1 = \
	      std::min(t1.hwPt(), 0x1FF) |
	      (t1.hwEtRatio() & 0x7) << 9 |
	      (t1.hwQual() & 0xF) << 12;

	    word1 = word1 |
	      std::min(t2.hwPt(), 0x1FF) << 16 |
	      (t2.hwEtRatio() & 0x7) << 25 |
	      (t2.hwQual() & 0xF) << 28;

            load.push_back(word1);

            // Do it all again for -eta

            uint32_t word2 = \
              std::min(t3.hwPt(), 0x1FF) |
              (t3.hwEtRatio() & 0x7) << 9 |
              (t3.hwQual() & 0xF) << 12;

            word2 = word2 |
              std::min(t4.hwPt(), 0x1FF) << 16 |
              (t4.hwEtRatio() & 0x7) << 25 |
              (t4.hwQual() & 0xF) << 28;

            load.push_back(word2);

          }

          res.push_back(Block(id, load));

        }
      }
      return res;
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::CaloTowerPacker);
