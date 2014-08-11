#include "DataFormats/L1Trigger/interface/Tau.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class TauPacker : public BasePacker {
      public:
         TauPacker(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual Blocks pack(const edm::Event&) override;
      private:
         edm::EDGetTokenT<TauBxCollection> tauToken_;
   };

   class TauPackerFactory : public BasePackerFactory {
      public:
         TauPackerFactory(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual PackerList create(const unsigned& fw, const int fedid) override;

      private:
         const edm::ParameterSet& cfg_;
         edm::ConsumesCollector& cc_;
   };
}

// Implementation

namespace l1t {
   TauPacker::TauPacker(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
   {
      tauToken_ = cc.consumes<TauBxCollection>(cfg.getParameter<edm::InputTag>("Taus"));
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
         int n = 0;
         for (auto j = taus->begin(i); j != taus->end(i) && n < 8; ++j, ++n) {
            uint32_t word = \
                            std::min(j->hwPt(), 0x1FF) |
                            (abs(j->hwEta()) & 0x7F) << 9 |
                            ((j->hwEta() < 0) & 0x1) << 16 |
                            (j->hwPhi() & 0xFF) << 17 |
                            (j->hwIso() & 0x1) << 25 |
                            (j->hwQual() & 0x7) << 26;
            res.load.push_back(word);
         }

         // pad for empty taus
         for (; n < 8; ++n)
            res.load.push_back(0);
      }

      return {res};
   }

   TauPackerFactory::TauPackerFactory(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) : cfg_(cfg), cc_(cc)
   {
   }

   PackerList
   TauPackerFactory::create(const unsigned& fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new TauPacker(cfg_, cc_))};
   }
}

DEFINE_L1TPACKER(l1t::TauPackerFactory);
