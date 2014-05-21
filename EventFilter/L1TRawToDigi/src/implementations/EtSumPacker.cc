#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class EtSumPacker : public BasePacker {
      public:
         EtSumPacker(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual Blocks pack(const edm::Event&) override;
      private:
         edm::EDGetToken etSumToken_;
   };

   class EtSumPackerFactory : public BasePackerFactory {
      public:
         EtSumPackerFactory(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual PackerList create(const unsigned& fw, const int fedid) override;

      private:
         const edm::ParameterSet& cfg_;
         edm::ConsumesCollector& cc_;
   };
}

// Implementation

namespace l1t {
   EtSumPacker::EtSumPacker(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
   {
      etSumToken_ = cc.consumes<EtSumBxCollection>(cfg.getParameter<edm::InputTag>("EtSums"));
   }

   Blocks
   EtSumPacker::pack(const edm::Event& event)
   {
      edm::Handle<EtSumBxCollection> etSums;
      event.getByToken(etSumToken_, etSums);

      // Return one block only
      Block res;
      res.id = 3;

      for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
         for (auto j = etSums->begin(i); j != etSums->end(i); ++j) {
	   uint32_t word = std::min(j->hwPt(), 0xFFF);
	   if ((j->getType()==l1t::EtSum::kMissingEt) || (j->getType()==l1t::EtSum::kMissingHt))
	     word = word | ((j->hwPhi() & 0xFF) << 12);
	   res.load.push_back(word);
         }
      }

      return {res};
   }

   EtSumPackerFactory::EtSumPackerFactory(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) : cfg_(cfg), cc_(cc)
   {
   }

   PackerList
   EtSumPackerFactory::create(const unsigned& fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new EtSumPacker(cfg_, cc_))};
   }
}

DEFINE_L1TPACKER(l1t::EtSumPackerFactory);
