#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/L1TDigiToRaw.h"

#include "EGammaPacker.h"

namespace l1t {
   class EGammaPacker : public BasePacker {
      public:
         EGammaPacker(const edm::ParameterSet&);
         virtual Blocks pack(const edm::Event&);
         virtual void fetchToken(L1TDigiToRaw*);
      private:
         edm::InputTag egTag_;
         edm::EDGetToken egToken_;
   };

   PackerList
   EGammaPackerFactory::create(const edm::ParameterSet& cfg, const FirmwareVersion& fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new EGammaPacker(cfg))};
   }

   EGammaPacker::EGammaPacker(const edm::ParameterSet& cfg) :
      egTag_(cfg.getParameter<edm::InputTag>("EGammas"))
   {
   }

   Blocks
   EGammaPacker::pack(const edm::Event& event)
   {
      edm::Handle<EGammaBxCollection> egs;
      event.getByToken(egToken_, egs);

      // Return one block only
      Block res;
      res.id = 1;

      for (int i = egs->getFirstBX(); i <= egs->getLastBX(); ++i) {
         for (auto j = egs->begin(i); j != egs->end(i); ++j) {
            uint32_t word = \
                            (j->hwPt() & 0x1FF) |
                            (abs(j->hwEta()) & 0x7F) << 9 |
                            ((j->hwEta() < 0) & 0x1) << 16 |
                            (j->hwPhi() & 0xFF) << 17 |
                            (j->hwIso() & 0x1) << 25 |
                            (j->hwQual() & 0x7) << 26;
            res.load.push_back(word);
         }
      }

      return {res};
   }

   void
   EGammaPacker::fetchToken(L1TDigiToRaw* digi2raw)
   {
      egToken_ = digi2raw->consumes<EGammaBxCollection>(egTag_);
   }
}
