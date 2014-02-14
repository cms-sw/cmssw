#ifndef DataFormats_L1Trigger_CaloCluster_h
#define DataFormats_L1Trigger_CaloCluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  
  class CaloCluster : public L1Candidate {
    public:
      enum ClusterFlag{
        PASS_THRES_SEED     = 0,
        PASS_FILTER_CLUSTER = 1,
        PASS_FILTER_NW      = 2,
        PASS_FILTER_N       = 3,
        PASS_FILTER_NE      = 4,
        PASS_FILTER_E       = 5,
        PASS_FILTER_SE      = 7,
        PASS_FILTER_S       = 8,
        PASS_FILTER_SW      = 9,
        PASS_FILTER_W       = 10,
        TRIM_LEFT           = 11,
        TRIM_RIGHT          = 12,
        EXT_UP              = 13,
        EXT_DOWN            = 14
      };

    public:
      CaloCluster(){}
      CaloCluster( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          );

      ~CaloCluster();

      void setClusterFlag(ClusterFlag flag, bool val=true);

      bool checkClusterFlag(ClusterFlag flag) const;
      int fgEta() const;
      int fgPhi() const;
      int hOverE() const;

    private:
      // Summary of clustering outcomes
      int m_clusterFlags; // see ClusterFlag bits (15 bits, will evolve)

      // fine grained position
      int m_fgEta; // 2 bits (to be defined in agreement with GT inputs)
      int m_fgPhi; // 2 bits (to be defined in agreement with GT inputs)

      // H/E
      int m_hOverE; // 7 bits (between 0 and 1 -> resolution=1/128=0.8%). Number of bits is not definitive
  };

  typedef BXVector<CaloCluster> CaloClusterBxCollection;
  
}

#endif
