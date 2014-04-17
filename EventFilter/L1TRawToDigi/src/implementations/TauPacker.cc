#include "DataFormats/L1Trigger/interface/Tau.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/L1TDigiToRaw.h"

#include "TauPacker.h"

namespace l1t {
   class TauPacker : public BasePacker {
      public:
         TauPacker(const edm::ParameterSet&);
         virtual Blocks pack(const edm::Event&);
         virtual void fetchToken(L1TDigiToRaw*);
      private:
         edm::InputTag tauTag_;
         edm::EDGetToken tauToken_;
   };

   PackerList
   TauPackerFactory::create(const edm::ParameterSet& cfg, const FirmwareVersion& fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new TauPacker(cfg))};
   }

   TauPacker::TauPacker(const edm::ParameterSet& cfg) :
      tauTag_(cfg.getParameter<edm::InputTag>("Taus"))
   {
   }

   Blocks
   TauPacker::pack(const edm::Event& event)
   {
      edm::Handle<TauBxCollection> taus;
      event.getByToken(tauToken_, taus);

      // Return one block only
      Block res;
      res.id = 7;

      for (int i = taus->getFirstBX(); i <= taus->getLastBX(); ++i) {
         for (auto j = taus->begin(i); j != taus->end(i); ++j) {
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
   TauPacker::fetchToken(L1TDigiToRaw* digi2raw)
   {
      tauToken_ = digi2raw->consumes<TauBxCollection>(tauTag_);
   }
}
