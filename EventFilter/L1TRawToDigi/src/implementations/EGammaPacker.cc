#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class EGammaPacker : public BasePacker {
      public:
         EGammaPacker(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual Blocks pack(const edm::Event&) override;
      private:
         edm::EDGetTokenT<EGammaBxCollection> egToken_;
   };

   class EGammaPackerFactory : public BasePackerFactory {
      public:
         EGammaPackerFactory(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual PackerList create(const unsigned& fw, const int fedid) override;

      private:
         const edm::ParameterSet& cfg_;
         edm::ConsumesCollector& cc_;
   };
}

// Implementation

namespace l1t {
   EGammaPacker::EGammaPacker(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
   {
      egToken_ = cc.consumes<EGammaBxCollection>(cfg.getParameter<edm::InputTag>("InputLabel"));
   }

   Blocks
   EGammaPacker::pack(const edm::Event& event)
   {
      edm::Handle<EGammaBxCollection> egs;
      event.getByToken(egToken_, egs);

      std::vector<uint32_t> load;

      for (int i = egs->getFirstBX(); i <= egs->getLastBX(); ++i) {
         int n = 0;
         for (auto j = egs->begin(i); j != egs->end(i) && n < 12; ++j, ++n) {
            uint32_t word = \
                            std::min(j->hwPt(), 0x1FF) |
                            (abs(j->hwEta()) & 0x7F) << 9 |
                            ((j->hwEta() < 0) & 0x1) << 16 |
                            (j->hwPhi() & 0xFF) << 17 |
                            (j->hwIso() & 0x1) << 25 |
                            (j->hwQual() & 0x7) << 26;
            load.push_back(word);
         }

         // pad for up to 12 egammas
         for (; n < 12; ++n)
            load.push_back(0);
      }

      return {Block(1, load)};
   }

   EGammaPackerFactory::EGammaPackerFactory(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) : cfg_(cfg), cc_(cc)
   {
   }

   PackerList
   EGammaPackerFactory::create(const unsigned& fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new EGammaPacker(cfg_, cc_))};
   }
}

DEFINE_L1TPACKER(l1t::EGammaPackerFactory);
