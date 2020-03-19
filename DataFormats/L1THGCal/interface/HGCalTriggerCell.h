#ifndef DataFormats_L1TCalorimeter_HGCalTriggerCell_h
#define DataFormats_L1TCalorimeter_HGCalTriggerCell_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace l1t {

  class HGCalTriggerCell;
  typedef BXVector<HGCalTriggerCell> HGCalTriggerCellBxCollection;

  class HGCalTriggerCell : public L1Candidate {
  public:
    HGCalTriggerCell() {}

    HGCalTriggerCell(const LorentzVector& p4, int pt = 0, int eta = 0, int phi = 0, int qual = 0, uint32_t detid = 0);

    ~HGCalTriggerCell() override;

    void setDetId(uint32_t detid) { detid_ = DetId(detid); }
    void setPosition(const GlobalPoint& position) { position_ = position; }

    uint32_t detId() const { return detid_.rawId(); }
    const GlobalPoint& position() const { return position_; }

    int subdetId() const { return detid_.subdetId(); }

    void setMipPt(double value) { mipPt_ = value; }
    double mipPt() const { return mipPt_; }

    void setUncompressedCharge(uint32_t value) { uncompressedCharge_ = value; }
    uint32_t uncompressedCharge() const { return uncompressedCharge_; }

    void setCompressedCharge(uint32_t value) { compressedCharge_ = value; }
    uint32_t compressedCharge() const { return compressedCharge_; }

    void setPt(double pT);

  private:
    DetId detid_;
    GlobalPoint position_;

    double mipPt_{0.};

    uint32_t uncompressedCharge_{0};
    uint32_t compressedCharge_{0};
  };

}  // namespace l1t

#endif
