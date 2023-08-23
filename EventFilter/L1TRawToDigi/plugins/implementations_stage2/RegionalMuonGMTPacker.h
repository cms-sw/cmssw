#ifndef L1T_PACKER_STAGE2_REGIONALMUONGMTPACKER_H
#define L1T_PACKER_STAGE2_REGIONALMUONGMTPACKER_H

#include <vector>
#include <map>
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"
#include "FWCore/Framework/interface/Event.h"

namespace l1t {
  namespace stage2 {
    class RegionalMuonGMTPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
      void setIsKbmtf() { isKbmtf_ = true; };
      void setUseEmtfDisplacementInfo() { useEmtfDisplacementInfo_ = true; };
      void setUseEmtfNominalTightShowers() { useEmtfNominalTightShowers_ = true; };
      void setUseEmtfLooseShowers() { useEmtfLooseShowers_ = true; };

    private:
      struct GMTObjects {
        std::vector<RegionalMuonCand> mus;
        RegionalMuonShower shower;
      };
      typedef std::map<size_t, std::map<size_t, GMTObjects>> GMTObjectMap;  // Map of BX --> linkID --> objects
      typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;
      void packTF(const GMTObjectMap& objMap,
                  int firstMuonBx,
                  int lastMuonBx,
                  int firstMuonShowerBx,
                  int lastMuonShowerBx,
                  Blocks&);
      std::pair<int, int> getMuons(GMTObjectMap& objMap,
                                   const edm::Event& event,
                                   const edm::EDGetTokenT<RegionalMuonCandBxCollection>& tfToken);
      std::pair<int, int> getMuonShowers(GMTObjectMap& objMap,
                                         const edm::Event& event,
                                         const edm::EDGetTokenT<RegionalMuonShowerBxCollection>& tfShowerToken);

      static constexpr size_t wordsPerBx_ = 6;  // number of 32 bit words per BX

      bool isKbmtf_{false};
      bool useEmtfDisplacementInfo_{false};
      bool useEmtfNominalTightShowers_{false};
      bool useEmtfLooseShowers_{false};
    };
  }  // namespace stage2
}  // namespace l1t

#endif
