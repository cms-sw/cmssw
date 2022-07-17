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
      struct GMTObjects {
        std::vector<Muon> mus;
        MuonShower shower;
      };
      typedef std::map<size_t, GMTObjects> GMTOutputObjectMap;  // Map of BX --> objects
      typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;

      std::pair<int, int> getMuonShowers(GMTOutputObjectMap& objMap,
                                         const edm::Event& event,
                                         const edm::EDGetTokenT<MuonShowerBxCollection>& showerToken);
      std::pair<int, int> getMuons(GMTOutputObjectMap& objMap,
                                   const edm::Event& event,
                                   const edm::EDGetTokenT<MuonBxCollection>& muonToken);
      void packBx(const GMTOutputObjectMap& objMap,
                  int firstMuonBx,
                  int lastMuonBx,
                  int firstMuonShowerBx,
                  int lastMuonShowerBx,
                  PayloadMap& payloadMap);

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
