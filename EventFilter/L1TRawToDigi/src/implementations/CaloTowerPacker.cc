#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CaloTowerPacker.h"

namespace l1t {
   class CaloTowerPacker : public BasePacker {
      public:
         CaloTowerPacker(const edm::ParameterSet&);
         virtual Blocks pack(const edm::Event&);
         virtual void fetchToken(L1TDigiToRaw*);
      private:
         edm::InputTag towers_;
   };

   PackerList
   CaloTowerPackerFactory::create(const edm::ParameterSet& cfg, unsigned fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new CaloTowerPacker(cfg))};
   }

   CaloTowerPacker::CaloTowerPacker(const edm::ParameterSet& cfg) :
      towers_(cfg.getParameter<edm::InputTag>("CaloTowers"))
   {
   }

   Blocks
   CaloTowerPacker::pack(const edm::Event&)
   {
      Blocks res;
      return res;
   }

   void
   CaloTowerPacker::fetchToken(L1TDigiToRaw*)
   {
   }
}
