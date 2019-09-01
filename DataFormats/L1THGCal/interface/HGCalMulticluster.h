#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1THGCal/interface/HGCalClusterT.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

namespace l1t {

  class HGCalMulticluster : public HGCalClusterT<l1t::HGCalCluster> {
  public:
    HGCalMulticluster() : hOverEValid_(false) {}
    HGCalMulticluster(const LorentzVector p4, int pt = 0, int eta = 0, int phi = 0);

    HGCalMulticluster(const edm::Ptr<l1t::HGCalCluster> &tc, float fraction = 1);

    ~HGCalMulticluster() override;

    float hOverE() const {
      // --- this below would be faster when reading old objects, as HoE will only be computed once,
      // --- but it may not be allowed by CMS rules because of the const_cast
      // --- and could potentially cause a data race
      // if (!hOverEValid_) (const_cast<HGCalMulticluster*>(this))->saveHOverE();
      // --- this below is safe in any case
      return hOverEValid_ ? hOverE_ : l1t::HGCalClusterT<l1t::HGCalCluster>::hOverE();
    }

    void saveHOverE() {
      hOverE_ = l1t::HGCalClusterT<l1t::HGCalCluster>::hOverE();
      hOverEValid_ = true;
    }

  private:
    float hOverE_;
    bool hOverEValid_;
  };

  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;

}  // namespace l1t

#endif
