#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

template<typename T, typename F>
l1t::Blocks
process(unsigned int id, const BXVector<T>& coll, F filter)
{
   std::vector<uint32_t> load;

   for (int i = coll.getFirstBX(); i <= coll.getLastBX(); ++i) {
      uint16_t jetbit[4] = {0, 0, 0, 0};
      int n = 0;
      for (auto j =  coll.begin(i); j != coll.end(i) && n < 4; ++j) {
         if (!filter(*j))
            continue;
         //std::cout << j->hwPt() << " @ " << j->hwEta() << ", " << j->hwPhi() << " > " << j->hwQual() << " > " << j->hwIso() << std::endl;
         jetbit[n++] = std::min(j->hwPt(), 0x3F) |
                     (abs(j->hwEta()) & 0x7) << 6 |
                     ((j->hwEta() >> 3) & 0x1) << 9 |
                     (j->hwPhi() & 0x1F) << 10;
      }
      uint32_t word0=(jetbit[0] & 0xFFFF) | ((jetbit[1] & 0xFFFF) << 16);
      uint32_t word1=(jetbit[2] & 0xFFFF) | ((jetbit[3] & 0xFFFF) << 16);
      
      word0 |= (1 << 31) | (1 << 15);
      word1 |= ((i == 0) << 31) | ((i == 0) << 15);

      load.push_back(word0);
      load.push_back(word1);
   }

   return {l1t::Block(id, load)};
}

namespace l1t {
  namespace stage1 {
    class IsoEGammaPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class NonIsoEGammaPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class CentralJetPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class ForwardJetPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class TauPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class IsoTauPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    Blocks
    IsoEGammaPacker::pack(const edm::Event& event, const PackerTokens* toks)
    {
       edm::Handle<EGammaBxCollection> egammas;
       event.getByToken(static_cast<const CaloTokens*>(toks)->getEGammaToken(), egammas);

       return process(1, *egammas, [](const l1t::EGamma& eg) -> bool { return eg.hwIso() == 1; });
    }

    Blocks
    NonIsoEGammaPacker::pack(const edm::Event& event, const PackerTokens* toks)
    {
       edm::Handle<EGammaBxCollection> egammas;
       event.getByToken(static_cast<const CaloTokens*>(toks)->getEGammaToken(), egammas);

       return process(2, *egammas, [](const l1t::EGamma& eg) -> bool { return eg.hwIso() == 0; });
    }

    Blocks
    CentralJetPacker::pack(const edm::Event& event, const PackerTokens* toks)
    {
       edm::Handle<JetBxCollection> jets;
       event.getByToken(static_cast<const CaloTokens*>(toks)->getJetToken(), jets);

       return process(3, *jets, [](const l1t::Jet& jet) -> bool { return !(jet.hwQual() & 2); });
    }

    Blocks
    ForwardJetPacker::pack(const edm::Event& event, const PackerTokens* toks)
    {
       edm::Handle<JetBxCollection> jets;
       event.getByToken(static_cast<const CaloTokens*>(toks)->getJetToken(), jets);

       return process(4, *jets, [](const l1t::Jet& jet) -> bool { return jet.hwQual() & 2; });
    }

    Blocks
    TauPacker::pack(const edm::Event& event, const PackerTokens* toks)
    {
       edm::Handle<TauBxCollection> taus;
       event.getByToken(static_cast<const CaloTokens*>(toks)->getTauToken(), taus);

       return process(5, *taus, [](const l1t::Tau& tau) -> bool { return true; });
    }

    Blocks
    IsoTauPacker::pack(const edm::Event& event, const PackerTokens* toks)
    {
       edm::Handle<TauBxCollection> taus;
       event.getByToken(static_cast<const CaloTokens*>(toks)->getIsoTauToken(), taus);

       return process(8, *taus, [](const l1t::Tau& tau) -> bool { return true; });
    }
  }
}

DEFINE_L1T_PACKER(l1t::stage1::IsoEGammaPacker);
DEFINE_L1T_PACKER(l1t::stage1::NonIsoEGammaPacker);
DEFINE_L1T_PACKER(l1t::stage1::CentralJetPacker);
DEFINE_L1T_PACKER(l1t::stage1::ForwardJetPacker);
DEFINE_L1T_PACKER(l1t::stage1::TauPacker);
DEFINE_L1T_PACKER(l1t::stage1::IsoTauPacker);
