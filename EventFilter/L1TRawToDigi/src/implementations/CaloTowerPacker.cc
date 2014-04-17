#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/L1TDigiToRaw.h"

#include "CaloTowerPacker.h"

namespace l1t {
   class CaloTowerPacker : public BasePacker {
      public:
         CaloTowerPacker(const edm::ParameterSet&);
         virtual Blocks pack(const edm::Event&);
         virtual void fetchToken(L1TDigiToRaw*);
      private:
         edm::InputTag towerTag_;
         edm::EDGetToken towerToken_;
   };

   PackerList
   CaloTowerPackerFactory::create(const edm::ParameterSet& cfg, unsigned fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new CaloTowerPacker(cfg))};
   }

   CaloTowerPacker::CaloTowerPacker(const edm::ParameterSet& cfg) :
      towerTag_(cfg.getParameter<edm::InputTag>("CaloTowers"))
   {
   }

   Blocks
   CaloTowerPacker::pack(const edm::Event& event)
   {

     // This is not correct as it is now. The towers have a block per two phi values so I need this to handle a vector of Blocks I think.

      edm::Handle<CaloTowerBxCollection> towers;
      event.getByToken(towerToken_, towers);

      Block res;

      for (int i = towers->getFirstBX(); i <= towers->getLastBX(); ++i) {
         for (auto j = towers->begin(i); j != towers->end(i); j++) {

	   // I think this loop relies on the time slices i.e. eta being in the correct order in the vector
	   // I need to change it to use the method to find the entry by eta and phi instead of the iterator here.
	   // #include L1Trigger / L1TCalorimeter / interface / CaloTools.h
	   // l1t::CaloTools::caloTowerHash(int iEta,int iPhi) returns an int, which is the index in the std::vector

	   // Need to check these two words are in the right order
            uint32_t word = \
	      (j->hwPt() & 0x1FF) |
	      (j->hwEtRatio() & 0x7) << 9 |
	      (j->hwQual() & 0xF) << 12;
	    
	    j++;

	    word = word |
	      (j->hwPt() & 0x1FF) << 16 |
	      (j->hwEtRatio() & 0x7) << 25 |
	      (j->hwQual() & 0xF) << 28;

	    res.id = 2*j->hwPhi() - 2;
            res.load.push_back(word);
         }
      }

      return {res};
   }

   void
   CaloTowerPacker::fetchToken(L1TDigiToRaw* digi2raw)
   {
      towerToken_ = digi2raw->consumes<CaloTowerBxCollection>(towerTag_);
   }
}
