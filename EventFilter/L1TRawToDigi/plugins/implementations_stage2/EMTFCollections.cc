#include "FWCore/Framework/interface/Event.h"
#include "EMTFCollections.h"

namespace l1t {
  namespace stage2 {
    EMTFCollections::~EMTFCollections() {
      // std::cout << "Inside EMTFCollections.cc: ~EMTFCollections" << std::endl;

      // Sort by processor to match uGMT unpacked order
      L1TMuonEndCap::sort_uGMT_muons(*regionalMuonCands_);

      // Apply ZeroSuppression: Only save RPC hits if there is at least one CSC LCT in the sector
      bool has_LCT[12] = {false};
      for (int iSect = 0; iSect < 12; iSect++) {
        for (const auto& h : *EMTFHits_) {
          if (h.Is_CSC() && h.Sector_idx() == iSect) {
            has_LCT[iSect] = true;
            break;
          }
        }
      }
      for (const auto& h : *EMTFHits_) {
        if (has_LCT[h.Sector_idx()] || (h.Is_RPC() == 0 && h.Is_GEM() == 0)) {
          EMTFHits_ZS_->push_back(h);
        }
      }
      for (const auto& c : *EMTFCPPFs_) {
        int sect_idx = c.emtf_sector() - 1 + 6 * (c.rpcId().region() == -1);
        if (has_LCT[sect_idx]) {
          EMTFCPPFs_ZS_->push_back(c);
        }
      }
      // TODO: Do we need something equivalent for GEMs - JS 03.07.20
      // for (const auto& g : *EMTFGEMPadClusters_) {
      //   int sect_idx = g.emtf_sector() - 1 + 6 * (g.gemId().region() == -1);
      //   if (has_LCT[sect_idx]) {
      //     EMTFGEMPadClusters_ZS_->push_back(g);
      //   }
      // }

      event_.put(std::move(regionalMuonCands_));
      event_.put(std::move(regionalMuonShowers_));
      event_.put(std::move(EMTFDaqOuts_));
      event_.put(std::move(EMTFHits_ZS_));
      event_.put(std::move(EMTFTracks_));
      event_.put(std::move(EMTFLCTs_));
      event_.put(std::move(EMTFCSCShowers_));
      event_.put(std::move(EMTFCPPFs_ZS_));
      event_.put(std::move(EMTFGEMPadClusters_));
      // event_.put(std::move(EMTFGEMPadClusters_ZS_));
    }
  }  // namespace stage2
}  // namespace l1t
