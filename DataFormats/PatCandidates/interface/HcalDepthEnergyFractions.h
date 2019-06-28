#ifndef DataFormats_PatCandidates_HcalDepthEnergyFractions_h
#define DataFormats_PatCandidates_HcalDepthEnergyFractions_h

#include <vector>
#include <cstdint>

namespace pat {

  //
  // Hcal depth energy fracrtion struct
  //
  class HcalDepthEnergyFractions {
  private:
    //do not store
    std::vector<float> fractions_;
    //store
    std::vector<uint8_t> fractionsI_;

  public:
    explicit HcalDepthEnergyFractions(const std::vector<float>& v) : fractions_(v), fractionsI_() { initUint8Vector(); }
    HcalDepthEnergyFractions() : fractions_(), fractionsI_() {}

    // produce vector of uint8 from vector of float
    void initUint8Vector() {
      fractionsI_.clear();
      for (auto frac : fractions_)
        fractionsI_.push_back((uint8_t)(frac * 200.));
    }

    // produce vector of float from vector of uint8_t
    void initFloatVector() {
      fractions_.clear();
      for (auto fracI : fractionsI_)
        fractions_.push_back(float(fracI) / 200.);
    }

    // reset vectors
    void reset(std::vector<float> v) {
      fractions_ = v;
      initUint8Vector();
    }

    // provide a full vector for each depth
    const std::vector<float>& fractions() const { return fractions_; }

    // provide info for individual depth
    float fraction(unsigned int i) const {
      if (i < fractions_.size())
        return fractions_[i];
      else
        return -1.;
    }

    // provide a full vector (in uint8_t) for each depth
    const std::vector<uint8_t>& fractionsI() const { return fractionsI_; }

    // provide info for individual depth (uint8_t)
    int fractionI(unsigned int i) const {
      if (i < fractionsI_.size())
        return int(fractionsI_[i]);
      else
        return -1;  // physical range 0-200
    }
  };

}  // namespace pat

#endif
