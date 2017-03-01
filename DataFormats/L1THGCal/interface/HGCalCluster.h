#ifndef DataFormats_L1Trigger_HGCalCluster_h
#define DataFormats_L1Trigger_HGCalCluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  
  class HGCalCluster : public L1Candidate {
    public:
      HGCalCluster(){}
      HGCalCluster( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          );

      ~HGCalCluster();

      void setHwPtEm  (uint32_t pt)    {hwPtEm_= pt;}
      void setHwPtHad (uint32_t pt)    {hwPtHad_ = pt;}
      void setHwSeedPt(uint32_t pt)    {hwSeedPt_ = pt;}
      void setSubDet  (uint32_t subdet){subDet_ = subdet;}
      void setLayer   (uint32_t layer) {layer_ = layer;}
      void setModule  (uint32_t module) {module_ = module;}
      void setHOverE  (uint32_t hOverE){hOverE_ = hOverE;}

      bool isValid()      const {return true;}
      uint32_t hwPtEm()   const {return hwPtEm_;}
      uint32_t hwPtHad()  const {return hwPtHad_;}
      uint32_t hwSeedPt() const {return hwSeedPt_;}

      uint32_t subDet() const {return subDet_;}
      uint32_t layer()  const {return layer_;}
      uint32_t module()  const {return module_;}

      uint32_t hOverE() const {return hOverE_;}

      bool operator<(const HGCalCluster& cl) const;
      bool operator>(const HGCalCluster& cl) const {return  cl<*this;};
      bool operator<=(const HGCalCluster& cl) const {return !(cl>*this);};
      bool operator>=(const HGCalCluster& cl) const {return !(cl<*this);};

    private:
      // Energies
      uint32_t hwPtEm_;
      uint32_t hwPtHad_;
      uint32_t hwSeedPt_;

      // HGC specific information
      uint32_t subDet_;
      uint32_t layer_;
      uint32_t module_;

      // identification variables
      uint32_t hOverE_; 
  };

  typedef BXVector<HGCalCluster> HGCalClusterBxCollection;

  
}

#endif
