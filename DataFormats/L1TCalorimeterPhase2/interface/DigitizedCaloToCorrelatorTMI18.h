#ifndef DataFormats_L1TCalorimeterPhase2_DigitizedCaloToCorrelatorTMI18_h
#define DataFormats_L1TCalorimeterPhase2_DigitizedCaloToCorrelatorTMI18_h

#include <ap_int.h>
#include <vector>
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"

namespace l1tp2 {

  typedef std::variant<std::monostate, l1tp2::GCTEmDigiCluster, l1tp2::GCTHadDigiCluster> GCTDigiCluster;
  typedef std::vector<l1tp2::GCTDigiCluster> GCTDigiClusterLink;

  class DigitizedCaloToCorrelatorTMI18 {
  private:
    // Data (to remove)
    ap_uint<64> CardData[162];

    GCTDigiClusterLink CardLink;

  public:
    DigitizedCaloToCorrelatorTMI18() {
      for (int i = 0; i < 162; i++) {
        CardData[i] = 0;
      }
    }
    DigitizedCaloToCorrelatorTMI18(ap_uint<64> data[162], GCTDigiClusterLink link) {
      for (int i = 0; i < 162; i++) {
        CardData[i] = data[i];
      }
      CardLink = link;
    }

    const ap_uint<64>* dataCard() const { return CardData; }
    const GCTDigiClusterLink& linkCard() const { return CardLink; }
  };

  // Collection typedef
  // this represents both the EM and PF clusters from a single GCT card in one link
  typedef std::vector<l1tp2::DigitizedCaloToCorrelatorTMI18> DigitizedCaloToCorrelatorCollectionTMI18;

}  // namespace l1tp2

#endif
