#include "DataFormats/L1Trigger/interface/Jet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/L1TDigiToRaw.h"

#include "JetPacker.h"

namespace l1t {
   class JetPacker : public BasePacker {
      public:
         JetPacker(const edm::ParameterSet&);
         virtual Blocks pack(const edm::Event&);
         virtual void fetchToken(L1TDigiToRaw*);
      private:
         edm::InputTag jetTag_;
         edm::EDGetToken jetToken_;
   };

   PackerList
   JetPackerFactory::create(const edm::ParameterSet& cfg, unsigned fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new JetPacker(cfg))};
   }

   JetPacker::JetPacker(const edm::ParameterSet& cfg) :
      jetTag_(cfg.getParameter<edm::InputTag>("Jets"))
   {
   }

   Blocks
   JetPacker::pack(const edm::Event& event)
   {
      edm::Handle<JetBxCollection> jets;
      event.getByToken(jetToken_, jets);

      // Return one block only
      Block res;
      res.id = 5;

      for (int i = jets->getFirstBX(); i <= jets->getLastBX(); ++i) {
         for (auto j = jets->begin(i); j != jets->end(i); ++j) {
            uint32_t word = \
                            (j->hwPt() & 0x7FF) |
                            (abs(j->hwEta()) & 0x7F) << 11 |
                            ((j->hwEta() < 0) & 0x1) << 18 |
                            (j->hwPhi() & 0xFF) << 19 |
                            (j->hwQual() & 0x7) << 27;
            res.load.push_back(word);
         }
      }

      return {res};
   }

   void
   JetPacker::fetchToken(L1TDigiToRaw* digi2raw)
   {
      jetToken_ = digi2raw->consumes<JetBxCollection>(jetTag_);
   }
}
