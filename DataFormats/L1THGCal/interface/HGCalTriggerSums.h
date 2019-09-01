#ifndef DataFormats_L1TCalorimeter_HGCalTriggerSums_h
#define DataFormats_L1TCalorimeter_HGCalTriggerSums_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace l1t {

  class HGCalTriggerSums;
  typedef BXVector<HGCalTriggerSums> HGCalTriggerSumsBxCollection;

  class HGCalTriggerSums : public L1Candidate {
  public:
    HGCalTriggerSums() {}

    HGCalTriggerSums(const LorentzVector& p4, int pt = 0, int eta = 0, int phi = 0, int qual = 0, uint32_t detid = 0);

    ~HGCalTriggerSums() override;

    void setDetId(uint32_t detid) { detid_ = DetId(detid); }
    void setPosition(const GlobalPoint& position) { position_ = position; }

    uint32_t detId() const { return detid_.rawId(); }
    const GlobalPoint& position() const { return position_; }

    void setMipPt(double value) { mipPt_ = value; }
    double mipPt() const { return mipPt_; }

  private:
    DetId detid_;
    GlobalPoint position_;

    double mipPt_;
  };

}  // namespace l1t

#endif
