#ifndef DataFormats_L1TCalorimeterPhase2_DigitizedCaloToCorrelatorTM18_h
#define DataFormats_L1TCalorimeterPhase2_DigitizedCaloToCorrelatorTM18_h

#include <ap_int.h>
#include <variant>
#include <vector>
#include <array>
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"

namespace l1tp2 {

  typedef std::variant<std::monostate, l1tp2::GCTEmDigiCluster, l1tp2::GCTHadDigiCluster> GCTDigiCluster;
  typedef std::vector<l1tp2::GCTDigiCluster> GCTDigiClusterLink;
  static constexpr int kNCardLinks = 162;
  static constexpr int EM_SLR1_POS_OFFSET = 1;
  static constexpr int EM_SLR1_NEG_OFFSET = 17;
  static constexpr int PF_SLR1_POS_OFFSET = 33;
  static constexpr int PF_SLR1_NEG_OFFSET = 57;
  static constexpr int EM_SLR3_POS_OFFSET = 82;
  static constexpr int EM_SLR3_NEG_OFFSET = 98;
  static constexpr int PF_SLR3_POS_OFFSET = 114;
  static constexpr int PF_SLR3_NEG_OFFSET = 138;

  class DigitizedCaloToCorrelatorTM18 {
  private:
    // Data
    std::array<ap_uint<64>, kNCardLinks> cardData_{};

    GCTDigiClusterLink cardLink_;

  public:
    DigitizedCaloToCorrelatorTM18() = default;
    DigitizedCaloToCorrelatorTM18(const std::array<ap_uint<64>, kNCardLinks>& data, const GCTDigiClusterLink link) {
      for (int i = 0; i < kNCardLinks; i++) {
        cardData_[i] = data[i];
      }
      cardLink_ = link;
    }

    const std::array<ap_uint<64>, kNCardLinks>& dataCard() const { return cardData_; }
    const GCTDigiClusterLink& linkCard() const { return cardLink_; }
  };

  // Collection typedef
  // this represents both the EM and PF clusters from a single GCT card in one link
  typedef std::vector<l1tp2::DigitizedCaloToCorrelatorTM18> DigitizedCaloToCorrelatorCollectionTM18;

}  // namespace l1tp2

#endif
