#ifndef DataFormats_L1TCalorimeterPhase2_GCTHadDigiCluster_h
#define DataFormats_L1TCalorimeterPhase2_GCTHadDigiCluster_h

#include <ap_int.h>
#include <vector>

#ifdef CMSSW_GIT_HASH
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloPFCluster.h"
#endif

namespace l1tp2 {

  class GCTHadDigiCluster {
  private:
    // Data
    unsigned long long int clusterData;

    // Constants
    static constexpr float LSB_PT = 0.5;  // 0.5 GeV

#ifdef CMSSW_GIT_HASH
    // reference to corresponding float cluster
    edm::Ref<l1tp2::CaloPFClusterCollection> clusterRef_;
#endif

  public:
    GCTHadDigiCluster() { clusterData = 0x0; }

    GCTHadDigiCluster(ap_uint<64> data) { clusterData = data; }

    GCTHadDigiCluster(
        ap_uint<12> pt, ap_uint<7> eta, ap_int<7> phi, ap_uint<12> ecal, ap_uint<6> fb, ap_uint<20> spare) {
      clusterData = ((ap_uint<64>)pt) | (((ap_uint<64>)eta) << 12) | (((ap_uint<64>)(phi & 0x7F)) << 19) |
                    (((ap_uint<64>)ecal) << 26) | (((ap_uint<64>)fb) << 38) | (((ap_uint<64>)spare << 44));
    }

#ifdef CMSSW_GIT_HASH
    // Setters
    void setRef(const edm::Ref<l1tp2::CaloPFClusterCollection>& clusterRef) { clusterRef_ = clusterRef; }
#endif

    // Getters
    ap_uint<64> data() const { return clusterData; }

    // Other getters
    float ptLSB() const { return LSB_PT; }
    ap_uint<12> pt() const { return (clusterData & 0xFFF); }
    float ptFloat() const { return pt() * ptLSB(); }

    // crystal eta (unsigned 7 bits)
    ap_uint<7> eta() const { return ((clusterData >> 12) & 0x7F); }

    // crystal phi (signed 7 bits)
    ap_int<7> phi() const { return ((clusterData >> 19) & 0x7F); }

    ap_uint<12> ecal() const { return ((clusterData >> 26) & 0xFFF); }

    ap_uint<6> fb() const { return ((clusterData >> 38) & 0x3F); }

    // Encoding region information
    ap_uint<20> spare() const { return ((clusterData >> 44) & 0xFFFFF); }

#ifdef CMSSW_GIT_HASH
    // Get the underlying ref
    const edm::Ref<l1tp2::CaloPFClusterCollection>& clusterRef() const { return clusterRef_; }
#endif
  };

  // Collection typedefs

  // This represents the 36 GCTHadDigiClusters in one link (one link spans 4 RCT cards, each RCT card sends 9 clusters (zero-padded and sorted by decreasing pT)
  // The ordering of the 4 RCT cards in this std::vector is, e.g. for GCT1.SLR3, real phi -50 to -20 degrees, then real phi -20 to 10 degrees, then real phi 10 to 40 degrees, and lastly real phi 40 to 70 degrees
  typedef std::vector<l1tp2::GCTHadDigiCluster> GCTHadDigiClusterLink;

  // This represents the 12 links sending GCTHadDigiClusters in the full barrel: there are 12 links = (3 GCT cards) * (two SLRs per GCT) * (one positive eta link and one negative eta link)
  // The ordering of the links in this std::vector is (GCT1.SLR1 negEta, GCT.SLR1 posEta, GCT1.SLR3 negEta, GCT1.SLR3 posEta, then analogously for GCT2 and GCT3)
  typedef std::vector<l1tp2::GCTHadDigiClusterLink> GCTHadDigiClusterCollection;

}  // namespace l1tp2

#endif
