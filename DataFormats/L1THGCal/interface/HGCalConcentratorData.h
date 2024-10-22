#ifndef DataFormats_L1TCalorimeter_HGCalConcentratorData_h
#define DataFormats_L1TCalorimeter_HGCalConcentratorData_h

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace l1t {

  class HGCalConcentratorData;
  typedef BXVector<HGCalConcentratorData> HGCalConcentratorDataBxCollection;

  class HGCalConcentratorData {
  public:
    HGCalConcentratorData(const uint32_t data = 0, uint32_t index = 0, uint32_t detid = 0);

    ~HGCalConcentratorData();

    void setDetId(uint32_t detid) { detid_ = DetId(detid); }

    uint32_t detId() const { return detid_.rawId(); }

    void setIndex(uint32_t value) { index_ = value; }
    uint32_t index() const { return index_; }

    void setData(uint32_t value) { data_ = value; }
    uint32_t data() const { return data_; }

  private:
    uint32_t data_{0};
    uint32_t index_{0};
    DetId detid_;
  };

}  // namespace l1t

#endif
