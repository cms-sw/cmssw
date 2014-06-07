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
        TRIM_NW             = 2,
        TRIM_N              = 3,
        TRIM_NE             = 4,
        TRIM_E              = 5,
        TRIM_SE             = 7,
        TRIM_S              = 8,
        TRIM_SW             = 9,
        TRIM_W              = 10,
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
      void setHwSeedPt(int pt);
      void setFgEta(int fgEta);
      void setFgPhi(int fgPhi);
      void setHOverE(int hOverE);
      void setFgECAL(int fgECAL);

      bool checkClusterFlag(ClusterFlag flag) const;
      bool isValid() const;
      int hwSeedPt() const;
      int fgEta() const;
      int fgPhi() const;
      int hOverE() const;
      int fgECAL() const;
      int clusterFlags() const{return m_clusterFlags;}

      bool operator<(const CaloCluster& cl) const;
      bool operator>(const CaloCluster& cl) const {return  cl<*this;};
      bool operator<=(const CaloCluster& cl) const {return !(cl>*this);};
      bool operator>=(const CaloCluster& cl) const {return !(cl<*this);};

    private:
      // Summary of clustering outcomes
      int m_clusterFlags; // see ClusterFlag bits (15 bits, will evolve)

      // Energies
      int m_hwSeedPt;

      // fine grained position
      int m_fgEta; // 2 bits (to be defined in agreement with GT inputs)
      int m_fgPhi; // 2 bits (to be defined in agreement with GT inputs)

      // identification variables
      int m_hOverE; // 8 bits (between 0 and 1 -> resolution=1/256=0.39%). Number of bits is not definitive
      int m_fgECAL; // FG bit of the seed tower
  };

  typedef BXVector<CaloCluster> CaloClusterBxCollection;

 
  //bool operator>(const CaloCluster& cll, const CaloCluster& clr) {return  operator< (clr,cll);};
  //bool operator<=(const CaloCluster& cll, const CaloCluster& clr) {return !operator> (cll,clr);};
  //bool operator>=(const CaloCluster& cll, const CaloCluster& clr) {return !operator< (cll,clr);};
  
}



#endif
