#ifndef DataFormats_L1Trigger_CaloCluster_h
#define DataFormats_L1Trigger_CaloCluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  namespace io_v1 {
    class CaloCluster : public L1Candidate {
    public:
      enum ClusterFlag {
        INCLUDE_SEED = 0,
        INCLUDE_NW = 1,
        INCLUDE_N = 2,
        INCLUDE_NE = 3,
        INCLUDE_E = 4,
        INCLUDE_SE = 5,
        INCLUDE_S = 6,
        INCLUDE_SW = 7,
        INCLUDE_W = 8,
        INCLUDE_NN = 9,
        INCLUDE_SS = 10,
        TRIM_LEFT = 11,
        IS_SECONDARY = 12,
        MERGE_UPDOWN = 13,    // 0=up, 1=down
        MERGE_LEFTRIGHT = 14  // 0=left, 1=right
      };

    public:
      CaloCluster() {}
      CaloCluster(const LorentzVector p4, int pt = 0, int eta = 0, int phi = 0);

      ~CaloCluster() override;

      void setClusterFlag(ClusterFlag flag, bool val = true);
      void setHwPtEm(int pt);
      void setHwPtHad(int pt);
      void setHwSeedPt(int pt);
      void setFgEta(int fgEta);
      void setFgPhi(int fgPhi);
      void setHOverE(int hOverE);
      void setFgECAL(int fgECAL);

      bool checkClusterFlag(ClusterFlag flag) const;
      bool isValid() const;
      int hwPtEm() const;
      int hwPtHad() const;
      int hwSeedPt() const;
      int fgEta() const;
      int fgPhi() const;
      int hOverE() const;
      int fgECAL() const;
      int clusterFlags() const { return m_clusterFlags; }

      bool operator<(const CaloCluster& cl) const;
      bool operator>(const CaloCluster& cl) const { return cl < *this; };
      bool operator<=(const CaloCluster& cl) const { return !(cl > *this); };
      bool operator>=(const CaloCluster& cl) const { return !(cl < *this); };

    private:
      // Summary of clustering outcomes
      int m_clusterFlags{0};  // see ClusterFlag bits (15 bits, will evolve)

      // Energies
      int m_hwPtEm{0};
      int m_hwPtHad{0};
      int m_hwSeedPt{0};

      // fine grained position
      int m_fgEta{0};  // 2 bits (to be defined in agreement with GT inputs)
      int m_fgPhi{0};  // 2 bits (to be defined in agreement with GT inputs)

      // identification variables
      int m_hOverE{0};  // 8 bits (between 0 and 1 -> resolution=1/256=0.39%). Number of bits is not definitive
      int m_fgECAL{0};  // FG bit of the seed tower
    };
  }  // namespace io_v1
  using CaloCluster = io_v1::CaloCluster;
  typedef BXVector<CaloCluster> CaloClusterBxCollection;

}  // namespace l1t

#endif
