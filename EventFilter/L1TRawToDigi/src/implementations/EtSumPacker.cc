#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/L1TDigiToRaw.h"

#include "EtSumPacker.h"

namespace l1t {
   class EtSumPacker : public BasePacker {
      public:
         EtSumPacker(const edm::ParameterSet&);
         virtual Blocks pack(const edm::Event&);
         virtual void fetchToken(L1TDigiToRaw*);
      private:
         edm::InputTag etSumTag_;
         edm::EDGetToken etSumToken_;
   };

   PackerList
   EtSumPackerFactory::create(const edm::ParameterSet& cfg, const FirmwareVersion& fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new EtSumPacker(cfg))};
   }

   EtSumPacker::EtSumPacker(const edm::ParameterSet& cfg) :
      etSumTag_(cfg.getParameter<edm::InputTag>("EtSums"))
   {
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
	   uint32_t word = (j->hwPt() & 0xFFF);
	   if ((j->getType()==l1t::EtSum::kMissingEt) || (j->getType()==l1t::EtSum::kMissingHt))
	     word = word | ((j->hwPhi() & 0xFF) << 12);
	   res.load.push_back(word);
         }
      }

      return {res};
   }

   void
   EtSumPacker::fetchToken(L1TDigiToRaw* digi2raw)
   {
      etSumToken_ = digi2raw->consumes<EtSumBxCollection>(etSumTag_);
   }
}
