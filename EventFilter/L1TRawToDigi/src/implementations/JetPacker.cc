#include "DataFormats/L1Trigger/interface/Jet.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class JetPacker : public BasePacker {
      public:
         JetPacker(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual Blocks pack(const edm::Event&) override;
      private:
         edm::EDGetTokenT<JetBxCollection> jetToken_;
   };

   class JetPackerFactory : public BasePackerFactory {
      public:
         JetPackerFactory(const edm::ParameterSet&, edm::ConsumesCollector&);
         virtual PackerList create(const unsigned& fw, const int fedid) override;

      private:
         const edm::ParameterSet& cfg_;
         edm::ConsumesCollector& cc_;
   };
}

// Implementation

namespace l1t {
   JetPacker::JetPacker(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
   {
      jetToken_ = cc.consumes<JetBxCollection>(cfg.getParameter<edm::InputTag>("Jets"));
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
         int n = 0;
         for (auto j = jets->begin(i); j != jets->end(i) && n < 12; ++j, ++n) {
            uint32_t word = \
                            std::min(j->hwPt(), 0x7FF) |
                            (abs(j->hwEta()) & 0x7F) << 11 |
                            ((j->hwEta() < 0) & 0x1) << 18 |
                            (j->hwPhi() & 0xFF) << 19 |
                            (j->hwQual() & 0x7) << 27;
            res.load.push_back(word);
         }

         for (; n < 12; ++n)
            res.load.push_back(0);
      }

      return {res};
   }

   JetPackerFactory::JetPackerFactory(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) : cfg_(cfg), cc_(cc)
   {
   }

   PackerList
   JetPackerFactory::create(const unsigned& fw, const int fedid)
   {
      return {std::shared_ptr<BasePacker>(new JetPacker(cfg_, cc_))};
   }
}

DEFINE_L1TPACKER(l1t::JetPackerFactory);
