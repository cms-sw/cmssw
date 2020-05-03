#ifndef L1T_PACKER_STAGE2_MUONPACKER_H
#define L1T_PACKER_STAGE2_MUONPACKER_H

#include <map>
#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"

namespace l1t {
  namespace stage2 {
    class MuonPacker : public Packer {
    public:
      MuonPacker(unsigned b1) : b1_(b1) {}
      Blocks pack(const edm::Event&, const PackerTokens*) override;
      unsigned b1_;
      inline void setFwVersion(unsigned fwId) { fwId_ = fwId; };
      inline void setFed(unsigned fedId) { fedId_ = fedId; };

    private:
      typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;

      void packBx(PayloadMap& payloadMap, const edm::Handle<MuonBxCollection>& muons, int bx);

      unsigned fwId_{0};
      unsigned fedId_{0};
    };

    class GTMuonPacker : public MuonPacker {
    public:
      GTMuonPacker() : MuonPacker(0) {}
    };
    class GMTMuonPacker : public MuonPacker {
    public:
      GMTMuonPacker() : MuonPacker(1) {}
    };
  }  // namespace stage2
}  // namespace l1t

#endif
